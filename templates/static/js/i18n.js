/**
 * Localisation: one catalogue, one lookup.
 *
 * **Chinese is the default.** A fresh install with no stored preference and a
 * browser reporting something the catalogue does not carry starts in Chinese,
 * because the product is Chinese-first and the backend's own prose is Chinese.
 * English is a first-class supported mode the reader can switch to, not a
 * consolation fallback: every key in this catalogue has both languages, and the
 * switcher is wired in `index.html`.
 *
 * **What this module is not.** It is not a document-language detector. The
 * corpus is Chinese and English papers, often mixed inside one file, and the
 * interface language says nothing about any of them — see `documentLocale()`
 * below, which is a *separate* axis and is never derived from the UI choice.
 *
 * **What the backend sends, and what is done with it.** `build_error_payload`
 * in `paperwhisperer/core/errors.py` puts a stable machine-readable `code`
 * next to a Chinese sentence on every failure. The `code` is the API and is
 * what every error presentation branches on; the sentence is prose whose
 * language is *not* assumed to match the reader's. `errorPresentation()`
 * therefore looks the code up in `ERROR_CATALOGUE` and only falls back to the
 * server's prose when the code is one this build has never heard of.
 */
import {
    ANSWER_MODE_DETAILS,
    ANSWER_MODE_LABELS,
    EXPORT_CITATIONS_HEADING,
    EXPORT_FACT_FIELD_LABELS,
    EXPORT_PRESENTATION_LABELS,
    EXPORT_PROVENANCE_HEADING,
    EXPORT_STATUS_LABELS,
    EXPORT_UNKNOWN_VALUE,
    HIGHLIGHT_ABSENT_NOTE,
    HIGHLIGHT_DRAWN_NOTE,
    NO_TARGET_MESSAGES,
    POSSIBLE_DUPLICATE_BASIS_FALLBACK,
    POSSIBLE_DUPLICATE_BADGE,
    POSSIBLE_DUPLICATE_HEADING,
    POSSIBLE_DUPLICATE_POLICY,
    RECOMMENDATION_CONFIDENCE_LABELS,
    RECOMMENDATION_NO_REASONS,
    RECOMMENDATION_REASONS_HEADING,
    RECOMMENDATION_SIGNAL_LABELS,
    RECOMMENDATION_SIGNAL_STATES,
    RECOMMENDATION_UNKNOWN_CONFIDENCE,
    REPARSE_CLOUD_WARNING,
    REPARSE_FORCE_OCR_WARNING,
    SECTION_EMPTY_TEXT,
    TARGET_CAPABILITY_LABELS,
    UNKNOWN_ERROR_PRESENTATION,
    VIEW_KIND_LABELS,
    VIEW_NOT_A_RASTER_BADGE,
    VIEW_TARGET_MISSING_MESSAGE
} from './config.js';

// ==========================================================================
//
// Locale identity
// ==========================================================================

/** The two locales this build ships. The first entry is not a preference. */
export const SUPPORTED_LOCALES = ['zh-CN', 'en'];

/**
 * Where every session starts that has not said otherwise.
 *
 * Chinese, deliberately. A reader who opens this product without choosing a
 * language is served Chinese, because the product is Chinese-first and the
 * backend's own error prose is Chinese — an English shell over Chinese errors
 * is a worse first impression than either language alone. Detection below
 * chooses the *starting* language; it can never move a Chinese-capable browser
 * to English, and it never overrides a choice the reader has already made.
 */
export const DEFAULT_LOCALE = 'zh-CN';

const STORAGE_KEY = 'paperwhisperer-locale';

let currentLocale = DEFAULT_LOCALE;
const localeListeners = new Set();

// ==========================================================================
//
// Detection
// ==========================================================================

/**
 * Map an arbitrary BCP-47-ish tag onto a locale this build ships, or `null`.
 *
 * The Chinese cases are the interesting ones and they are handled on purpose
 * rather than by accident. This product's Chinese is **simplified**, and it is
 * the only Chinese it ships — there is no traditional catalogue to fall back
 * to. So a `zh-TW` / `zh-Hant` preference resolves to `zh-CN`: the reader is
 * reading a UI whose *interface* we do have, and the interface being simplified
 * is the lesser surprise compared with a reader who asked for a language and
 * got a blank or a fallback box. It also means `zh-Hans-CN`, `zh-Hans`,
 * `zh-SG` and `zh` all land in the same place, which is what a reader expects
 * from any of them.
 *
 * Anything unrecognised returns `null` rather than guessing, so the caller can
 * fall back to {@link DEFAULT_LOCALE} and record that it did.
 */
export function normalizeLocale(raw) {
    const value = String(raw ?? '').trim().toLowerCase().replace(/_/g, '-');
    if (!value) return null;

    // A regional variant is the interesting prefix: `en-gb`, `zh-hans-cn`.
    if (value === 'zh' || value.startsWith('zh-')) {
        // Includes zh-TW, zh-HK, zh-MO, zh-Hant, zh-Hans and every combination:
        // simplified is the only Chinese this catalogue carries.
        return 'zh-CN';
    }
    if (value === 'en' || value.startsWith('en-')) {
        return 'en';
    }
    return null;
}

/**
 * What the browser says it would prefer, as a supported locale or `null`.
 *
 * Every failure mode — no `navigator`, no `languages`, an empty array, a
 * `navigator` that throws on property access (some embedded webviews do), a
 * tag this build does not carry — returns `null` so that the caller applies
 * the default. Detection is a convenience, never a hard dependency.
 */
export function detectLocale() {
    try {
        const nav = typeof navigator === 'undefined' ? null : navigator;
        if (!nav) return null;

        const reported = Array.isArray(nav.languages) && nav.languages.length
            ? nav.languages
            : (nav.language ? [nav.language] : []);

        for (const tag of reported) {
            const normalized = normalizeLocale(tag);
            if (normalized) return normalized;
        }
        return null;
    } catch {
        return null;
    }
}

// ==========================================================================
//
// Persistence
// ==========================================================================

/**
 * `localStorage`, or `null` when it is not usable.
 *
 * A bare `localStorage.getItem` throws outright in private-browsing modes and
 * whenever storage is disabled by policy, and the property access itself
 * throws in a sandboxed frame. A language switcher that takes the interface
 * down when storage is unavailable is a worse defect than one that forgets a
 * preference, so every access is guarded and degrades to "no preference" —
 * which lands on the default and on detection, both of which work without
 * storage.
 */
function safeStorage() {
    try {
        const storage = typeof localStorage === 'undefined' ? null : localStorage;
        if (!storage) return null;
        // Touching the API is the only reliable probe: Safari's private mode
        // exposes the object and throws on write.
        const probe = '__pw_locale_probe__';
        storage.setItem(probe, '1');
        storage.removeItem(probe);
        return storage;
    } catch {
        return null;
    }
}

function readStoredLocale() {
    const storage = safeStorage();
    if (!storage) return null;
    try {
        const raw = storage.getItem(STORAGE_KEY);
        if (raw === null) return null;
        // A stored value is only honoured when it names a locale this build
        // actually ships. A preference stored wrongly — by an older build, a
        // hand-edited value, a half-written write — is discarded rather than
        // obeyed, because a wrong preference is worse than no preference.
        return normalizeLocale(raw);
    } catch {
        return null;
    }
}

function writeStoredLocale(locale) {
    const storage = safeStorage();
    if (!storage) return false;
    try {
        storage.setItem(STORAGE_KEY, locale);
        return true;
    } catch {
        return false;
    }
}

/**
 * The locale to start in: an explicit stored choice, else the browser's
 * preference, else {@link DEFAULT_LOCALE}.
 *
 * Order is the whole point. A reader who has chosen must keep their choice
 * across a reload, so the stored value is consulted first and is never
 * second-guessed by detection.
 */
export function resolveInitialLocale() {
    return readStoredLocale() || detectLocale() || DEFAULT_LOCALE;
}

// ==========================================================================
//
// Lookup
// ==========================================================================

/**
 * The `zh-CN` strings, keyed the same way as every other locale.
 *
 * The Chinese values that already existed as module-level constants are
 * *reused* rather than retyped: `config.js` holds them, and this object points
 * at them. That keeps one copy of each sentence in the codebase, so a wording
 * fix in `config.js` cannot leave a stale duplicate behind here. The English
 * blocks that were sitting in `config.js` — `SECTION_EMPTY_TEXT`,
 * `ANSWER_MODE_LABELS` and the rest — are the *English* locale's values and
 * live in the `en` table below; the Chinese values for those keys are new.
 */
const ZH_CN = {
    // --- section placeholders ---------------------------------------------
    'section.empty.summary': '摘要会在分析完成后出现在这里。',
    'section.empty.quotes': '关键引用会在分析完成后出现在这里。',
    'section.empty.mindmap': '文本结构会在分析完成后出现在这里。',
    'section.empty.evaluation': '启用批判性评估后，评估结果会出现在这里。',
    'section.empty.research_brief': '启用深度研究简报后，简报会出现在这里。',

    // --- answer modes -----------------------------------------------------
    'mode.evidence': '证据',
    'mode.explain': '讲解',
    'mode.critique': '批判',
    'mode.reproduce': '复现',
    'mode.detail.evidence': '证据模式先给出答案，再引用文档中的支撑与不确定性。',
    'mode.detail.explain': '讲解模式逐步讲解概念、方法与公式。',
    'mode.detail.critique': '批判模式审视优点、假设、局限与效度威胁。',
    'mode.detail.reproduce': '复现模式把论文拆成步骤、变量、依赖与风险。',

    // --- reparse warnings -------------------------------------------------
    'reparse.force_ocr_warning': REPARSE_FORCE_OCR_WARNING,
    'reparse.cloud_warning': REPARSE_CLOUD_WARNING,

    // --- evidence targeting ------------------------------------------------
    'evidence.no_target.unknown_locator': NO_TARGET_MESSAGES.unknown_locator,
    'evidence.no_target.relative_bbox': NO_TARGET_MESSAGES.relative_bbox,
    'evidence.no_target.missing_bbox': NO_TARGET_MESSAGES.missing_bbox,
    'evidence.no_target.missing_page': NO_TARGET_MESSAGES.missing_page,
    'evidence.no_target.malformed_bbox': NO_TARGET_MESSAGES.malformed_bbox,
    'evidence.no_target.incomplete_locator': NO_TARGET_MESSAGES.incomplete_locator,
    'evidence.no_target.fallback': '这条证据没有可用的视觉定位。',
    'evidence.capability.pdf_page_highlight': TARGET_CAPABILITY_LABELS.pdf_page_highlight,
    'evidence.capability.slide_jump': TARGET_CAPABILITY_LABELS.slide_jump,
    'evidence.capability.sheet_cells': TARGET_CAPABILITY_LABELS.sheet_cells,
    'evidence.capability.element_jump': TARGET_CAPABILITY_LABELS.element_jump,
    'evidence.capability.offset_jump': TARGET_CAPABILITY_LABELS.offset_jump,
    'evidence.view_kind.page_map': VIEW_KIND_LABELS.page_map,
    'evidence.view_kind.slide_view': VIEW_KIND_LABELS.slide_view,
    'evidence.view_kind.element_view': VIEW_KIND_LABELS.element_view,
    'evidence.view_kind.fallback': '定位视图',
    'evidence.view_target_missing': VIEW_TARGET_MISSING_MESSAGE,
    'evidence.highlight_drawn': HIGHLIGHT_DRAWN_NOTE,
    'evidence.highlight_absent': HIGHLIGHT_ABSENT_NOTE,
    'evidence.not_a_raster': VIEW_NOT_A_RASTER_BADGE,
    'evidence.unsupported_view': '这种定位类型没有可展示的视图。',

    // --- recommendations ---------------------------------------------------
    'recommend.signal.cited_by': RECOMMENDATION_SIGNAL_LABELS.cited_by,
    'recommend.signal.citing': RECOMMENDATION_SIGNAL_LABELS.citing,
    'recommend.signal.related': RECOMMENDATION_SIGNAL_LABELS.related,
    'recommend.signal.topical': RECOMMENDATION_SIGNAL_LABELS.topical,
    'recommend.signal.same_dataset': RECOMMENDATION_SIGNAL_LABELS.same_dataset,
    'recommend.signal.fallback': '信号',
    'recommend.state.ok.label': RECOMMENDATION_SIGNAL_STATES.ok.label,
    'recommend.state.ok.summary': RECOMMENDATION_SIGNAL_STATES.ok.summary,
    'recommend.state.empty.label': RECOMMENDATION_SIGNAL_STATES.empty.label,
    'recommend.state.empty.summary': RECOMMENDATION_SIGNAL_STATES.empty.summary,
    'recommend.state.unavailable.label': RECOMMENDATION_SIGNAL_STATES.unavailable.label,
    'recommend.state.unavailable.summary': RECOMMENDATION_SIGNAL_STATES.unavailable.summary,
    'recommend.state.skipped.label': RECOMMENDATION_SIGNAL_STATES.skipped.label,
    'recommend.state.skipped.summary': RECOMMENDATION_SIGNAL_STATES.skipped.summary,
    'recommend.confidence.declared': RECOMMENDATION_CONFIDENCE_LABELS.declared,
    'recommend.confidence.derived': RECOMMENDATION_CONFIDENCE_LABELS.derived,
    'recommend.confidence.unknown': RECOMMENDATION_UNKNOWN_CONFIDENCE,
    'recommend.reasons_heading': RECOMMENDATION_REASONS_HEADING,
    'recommend.no_reasons': RECOMMENDATION_NO_REASONS,
    'recommend.duplicate.heading': POSSIBLE_DUPLICATE_HEADING,
    'recommend.duplicate.policy': POSSIBLE_DUPLICATE_POLICY,
    'recommend.duplicate.badge': POSSIBLE_DUPLICATE_BADGE,
    'recommend.duplicate.basis_fallback': POSSIBLE_DUPLICATE_BASIS_FALLBACK,
    'recommend.duplicate.untitled': '（未命名结果）',
    'recommend.signals_heading': '推荐信号状态',
    'recommend.paper_count': '{count} 篇',

    // --- export -------------------------------------------------------------
    'export.citations_heading': EXPORT_CITATIONS_HEADING,
    'export.provenance_heading': EXPORT_PROVENANCE_HEADING,
    'export.unknown_value': EXPORT_UNKNOWN_VALUE,
    'export.presentation.direct_quote': EXPORT_PRESENTATION_LABELS.direct_quote,
    'export.presentation.paraphrase': EXPORT_PRESENTATION_LABELS.paraphrase,
    'export.fact.research_question': EXPORT_FACT_FIELD_LABELS.research_question,
    'export.fact.contributions': EXPORT_FACT_FIELD_LABELS.contributions,
    'export.fact.method': EXPORT_FACT_FIELD_LABELS.method,
    'export.fact.datasets': EXPORT_FACT_FIELD_LABELS.datasets,
    'export.fact.metrics': EXPORT_FACT_FIELD_LABELS.metrics,
    'export.fact.main_results': EXPORT_FACT_FIELD_LABELS.main_results,
    'export.fact.limitations': EXPORT_FACT_FIELD_LABELS.limitations,
    'export.fact.reproducibility': EXPORT_FACT_FIELD_LABELS.reproducibility,
    'export.fact.unnamed': '未命名字段',
    'export.status.extracted': EXPORT_STATUS_LABELS.extracted,
    'export.status.needs_review': EXPORT_STATUS_LABELS.needs_review,
    'export.status.not_found': EXPORT_STATUS_LABELS.not_found,

    // --- the unrecognised code ---------------------------------------------
    // Shown when the server sends a `code` this build has no entry for.
    'error.unknown.title': UNKNOWN_ERROR_PRESENTATION.title,
    'error.unknown.message': UNKNOWN_ERROR_PRESENTATION.message,
    'error.unknown.hint': UNKNOWN_ERROR_PRESENTATION.hint,

    // --- the switcher itself -----------------------------------------------
    'switcher.group_label': '语言 / Language',

    // --- static markup: skip links and header ------------------------------
    'skip.to_upload': '跳到文档上传',
    'skip.to_search': '跳到论文检索',
    'skip.to_result': '跳到分析工作区',
    'brand.subtitle': '研究工作台',
    'hero.copy': '上传一篇论文，抽取结构、生成结构图，并在同一个工作台里持续追问。这个页面现在更像一张研究桌，而不是一次性表单。',
    'hero.stats.types': '文档类型',
    'hero.stats.streaming': '章节流式输出',
    'hero.stats.followups': '智能追问',
    'nav.upload': '上传',
    'nav.paper_search': '论文检索',
    'nav.reading_queue': '阅读队列',
    'nav.analysis': '分析工作台',
    'status.waiting': '等待文档',

    // --- static markup: upload workspace ------------------------------------
    'upload.workspace_title': '文档上传工作区',
    'upload.intro': '支持 PDF、TXT、DOCX 和 PPTX 文件。一次成功的分析会替换当前会话；失败的尝试会保留上一次已完成的工作台。',
    'upload.shortcut_hints': '工作区快捷键',
    'upload.api_key_label': 'API Key（可选）',
    'upload.api_key_placeholder': '请输入你的 API Key',
    'upload.source_label': '文档来源',
    'upload.dropzone_label': '选择或拖放一个文档文件',
    'upload.dropzone_title': '把论文拖到这里，或点击浏览',
    'upload.dropzone_hint': 'PDF、TXT、DOCX 或 PPTX · 最大 16 MB',
    'upload.file_meta_empty': '尚未选择文件。建议使用结构提取效果更好的 PDF、TXT、DOCX 或 PPTX。',
    'upload.opt_mermaid': '启用结构图',
    'upload.opt_evaluation': '启用批判性评估',
    'upload.opt_brief': '启用研究简报',
    'upload.analyze': '分析文档',
    'upload.cancel_analysis': '取消分析',
    'upload.preparing': '正在准备文档分析…',
    'progress.upload': '上传',
    'progress.analyze': '分析',
    'progress.render': '渲染',
    'progress.ready': '就绪',
    'progress.label': '分析进度',

    // --- static markup: search, queue, export, results ---------------------
    'search.title': '论文检索',
    'search.query_label': '论文检索关键词',
    'search.placeholder': '按主题、任务、方法或数据集检索…',
    'search.submit': '检索论文',
    'search.cancel': '取消检索',
    'search.meta': '用一个查询同时检索 Semantic Scholar 与 arXiv。',
    'search.examples_label': '论文检索示例',
    'queue.title': '阅读队列',
    'queue.clear': '清空',
    'queue.meta': '把检索或推荐中的论文保存到当前会话的阅读队列。',
    'parse.title': '解析状态',
    'parse.reparse': '重新解析',
    'parse.unknown_notice': '服务端没有说明这份文档是怎么解析的。',
    'parse.cloud_result': '云端解析结果',
    'parse.cloud_outcome.success': '云端解析完成，返回了文本。',
    'parse.cloud_outcome.partial': '云端解析只返回了部分内容，这份文档可能不完整。',
    'parse.cloud_outcome.quota_exhausted': '云端解析额度已用尽，这次没有得到云端文本。',
    'parse.cloud_outcome.timed_out': '云端解析在拿到结果之前停止了等待。',
    'parse.cloud_outcome.blocked': '策略拒绝把这份文档交给云端解析。',
    'parse.cloud_outcome.failed': '云端解析没有返回可用的结果。',
    'parse.cloud_outcome.unknown': '服务端报告了一个本版本不认识的云端结果。',
    'parse.fieldset_legend': '重新解析这份文档',
    'parse.parser_label': '解析器',
    'parse.force_ocr': '重跑解析（当前解析器已默认开启 OCR）',
    'parse.submit': '立即重新解析',
    'export.title': '会话导出',
    'export.format_label': '格式',
    'export.submit': '导出会话',
    'export.hint': '导出件会带上每条引用的证据链接、文档版本与分析元数据（模型、提示词版本、证据版本、成本），这样几个月后再打开这个文件，仍然能说清它是怎么生成的。',
    'export.preview_empty': '成功运行后，可以导出当前分析、问答记录、研究轨迹与 Mermaid 资源。',
    'result.overview': '概览',
    'result.citations': '关键引用',
    'result.structure': '文本结构',
    'result.visual_map': '可视化结构图',
    'result.evaluation': '批判性评估',
    'result.brief': '深度研究简报',
    'result.recommendations': '自动推荐',
    'result.recommend_submit': '从当前论文生成推荐',
    'result.recommend_meta': '先分析一篇论文，然后从当前会话生成后续阅读建议。',
    'result.signals_label': '推荐信号状态',
    'result.ask': '提问',
    'result.ask_label': '关于当前文档的问题',
    'result.ask_placeholder': '询问证据、方法、局限或可复现性…',
    'result.ask_send': '发送',
    'result.ask_cancel': '取消',
    'result.answer_mode_label': '回答模式',
    'result.starter_label': '入门问题',
    'result.smart_prompts_label': '建议的后续问题',
    'result.next_actions_label': '推荐的行动',
    'result.copy': '复制',
    'result.recommend_cancel': '取消',

    // --- static markup: footer and evidence panel --------------------------
    'footer.ai_label': 'AI 协作者',
    'footer.crafted_with': '由',
    'evidence.heading': '证据',
    'evidence.close': '关闭',
    'evidence.close_label': '关闭证据面板',
    'evidence.idle_title': '尚未打开引用。',
    'evidence.idle_detail': '选择一枚引用标签，就能看到某条结论背后的原文片段。你的问题和阅读位置都会被保留。',
    'evidence.loading_title': '正在打开证据…',
    'evidence.error_title': '无法打开这条证据。',
    'evidence.no_target_title': '这条证据没有可用的视觉定位。',
    'evidence.no_target_detail': '下方仍然显示原文片段。',
    'evidence.empty_title': '这条证据有结果，但没有文本。',
    'evidence.empty_detail': '这条引用是真实的，只是它指向的原文片段是空的。',
    'evidence.truncated': '片段过长，仅显示前 8000 个字符',
    'evidence.characters': '字符 {start}–{end}',
    'evidence.page_images_unavailable': '这个版本不渲染页面图片。',
    'evidence.table_shell_empty': '解析器只返回了一个空表格骨架；下面的单元格没有恢复出内容。',
    'evidence.reason_code': '原因码：{code}',
    'evidence.still_shown': '下方仍然显示原文片段。',
    'evidence.no_passage': '这条引用也没有原文文本。',
    'evidence.opening': '正在打开证据…',
    'evidence.opened': '已打开证据',
    'evidence.closed': '已关闭证据',
    'evidence.unavailable': '证据不可用',
    'evidence.failed': '无法打开这条证据。',
    'evidence.no_document_id': '当前会话没有文档 id，因此无法定位它的证据。',
    'evidence.id_not_published': '这条引用是真实的，但服务端还没有公布它所属的文档 id。',
    'evidence.citations_label': '引用',

    // --- claims.js: the reader-facing half of the release gate ---------------
    //
    // These three prefixes are the only thing standing between a verified
    // quotation and a model paraphrase on the screen, so they are translated
    // rather than concatenated: a reader who cannot read the sentence that
    // says "this was not verified" cannot be warned at all. Each language says
    // the same thing about the same claim — nothing here decides the
    // presentation, which the backend already did.
    'claim.prefix.direct_quote': '逐字引用：',
    'claim.prefix.normalized_quote': '归一化后逐字对齐：',
    'claim.prefix.paraphrase': '转述（未通过逐字回源校验）：',
    'claim.chip.direct_quote': '已回源校验的直接引用{label}，打开证据',
    'claim.chip.paraphrase': '未通过回源校验的转述，不提供引用定位',
    'claim.chips_group_label': '打开证据',
    'claim.chip_generic_label': '证据',

    // --- the evidence panel's own sentences, produced at render time ---------
    // The state banners above carry `data-i18n` and are translated by
    // `translateStaticMarkup`; these four are written into elements that carry
    // no `data-i18n`, so nothing translated them before this table existed.
    'evidence.page_label': '第 {label} 页',
    'evidence.snippet.citation': '引用 {label}',
    'evidence.snippet.verbatim': '来自已核对的原文',
    'evidence.snippet.truncated': '片段已截断',

    // --- the parse card's field labels, notices and quality verdict ---------
    //
    // The card already re-rendered on a language switch; it read its labels
    // and its notices from literals, so a re-render reproduced the same
    // English in whichever language the reader had chosen. Saying the same
    // thing in the reader's language is the whole of this block.
    'parse.field.parser': '解析器',
    'parse.field.media_type': '媒体类型',
    'parse.field.status': '解析状态',
    'parse.field.cache': '缓存',
    'parse.value.not_reported': '未报告',
    'parse.value.cache_hit': '本次命中缓存，没有运行解析器。',
    'parse.notice.used_cloud': '这份文档被上传到了云端解析服务，内容离开了这台机器。',
    'parse.notice.cloud_discarded': '调用了云端解析并产生了费用，但结果被丢弃；你看到的文本来自本地解析。',
    'parse.notice.local_only': '没有解析器访问网络，这份文档一直留在这台机器上。',
    'parse.notice.fallback': '第一个解析器没有产出可用结果；这份结果来自回退解析器。',
    'parse.notice.failed': '解析失败：{message}',
    'parse.quality.absent': '这份文档没有返回解析质量报告。',
    'parse.quality.heading': '解析质量：{status}',
    'parse.quality.clean': '已检查的规则全部通过。',
    'parse.quality.rule_fallback': '规则',

    // --- the reparse control -----------------------------------------------
    //
    // The parser menu labels each choice with what it *requires*, not merely
    // what it is called, because `PARSER_BACKEND=legacy` (the shipped default)
    // makes every named parser answer `409 parser_unavailable` — and the
    // default route is chosen per media type, so there is no single "best
    // parser" the menu could have pointed at instead. See
    // `REPARSE_PARSER_OPTIONS` for why the options are labelled rather than
    // filtered.
    'reparse.parser.default': '按部署配置的解析链（按文件类型选择）',
    'reparse.parser.docling': 'Docling（本地）',
    'reparse.parser.docling_ocr': 'Docling + 强制 OCR（本地）',
    'reparse.parser.mineru_cloud': 'MinerU 云端解析',
    'reparse.parser.default_note': '这一项不指定解析器：部署会按当前文件的类型选路，不同文件类型可能走不同的解析器。',
    'reparse.parser.requires_gateway': '需要管理员把 PARSER_BACKEND 设为 gateway，否则服务器会以 409 parser_unavailable 拒绝这次请求。',
    'reparse.parser.requires_gateway_ocr': '需要管理员把 PARSER_BACKEND 设为 gateway，否则服务器会以 409 parser_unavailable 拒绝这次请求；强制 OCR 由所选的 Docling 自己执行。',
    'reparse.parser.requires_cloud': '需要部署配置云端解析适配器、隐私策略允许出网，并且还有可用调用额度。',
    'reparse.hint.ready': '用当前选中的选项重新解析这份文档。',
    'reparse.hint.no_document': '当前会话没有文档 id，因此没有可重新解析的对象。',
    'reparse.result.started': '正在重新解析…',
    'reparse.result.finished': '重新解析完成。上方的解析状态反映的是这次新的运行。',
    'reparse.result.no_document': '这次会话没有文档 id，因此没有可重新解析的对象。',
    'reparse.result.no_route': '这个服务端没有提供重新解析接口（POST {path} → {reason}）。文档没有被改动。',
    'reparse.result.failed': '重新解析请求没有成功（{reason}）。文档没有被改动。',
    'reparse.status.started': '正在重新解析文档…',
    'reparse.status.finished': '重新解析完成',
    'reparse.status.unavailable': '重新解析不可用',

    // --- leaf modules: shortcuts, theme, mermaid, empty content -------------
    'a11y.shortcut_upload': '上传快捷键已聚焦文档来源',
    'a11y.shortcut_search': '检索快捷键已聚焦论文检索',
    'a11y.shortcut_ask': '提问快捷键已聚焦问题输入框',
    'theme.switch_to_light': '切换到浅色主题',
    'theme.switch_to_dark': '切换到深色主题',
    'mermaid.failed_to_load': 'Mermaid 加载失败。',
    'mermaid.too_complex': '结构过于复杂，无法渲染。原始数据回退：',
    'content.none_available': '没有可显示的内容。',

    // --- dom.js: the legacy string path and the AI list toggle --------------
    'error.request_failed': '请求失败。',
    'footer.hide_ai': '隐藏 AI 协作者',
    'footer.show_ai': '显示 AI 协作者',

    // --- dom.js: copy feedback ---------------------------------------------
    'copy.no_content': '没有内容',
    'copy.nothing_yet': '还没有可复制的内容',
    'copy.done': '已复制',
    'copy.done_status': '内容已复制',
    'copy.failed': '复制失败',
    'copy.failed_status': '复制失败'
};

const EN = {
    'section.empty.summary': SECTION_EMPTY_TEXT.summary,
    'section.empty.quotes': SECTION_EMPTY_TEXT.quotes,
    'section.empty.mindmap': SECTION_EMPTY_TEXT.mindmap,
    'section.empty.evaluation': SECTION_EMPTY_TEXT.evaluation,
    'section.empty.research_brief': SECTION_EMPTY_TEXT.research_brief,

    'mode.evidence': ANSWER_MODE_LABELS.evidence,
    'mode.explain': ANSWER_MODE_LABELS.explain,
    'mode.critique': ANSWER_MODE_LABELS.critique,
    'mode.reproduce': ANSWER_MODE_LABELS.reproduce,
    'mode.detail.evidence': ANSWER_MODE_DETAILS.evidence,
    'mode.detail.explain': ANSWER_MODE_DETAILS.explain,
    'mode.detail.critique': ANSWER_MODE_DETAILS.critique,
    'mode.detail.reproduce': ANSWER_MODE_DETAILS.reproduce,

    'reparse.force_ocr_warning': 'Forced OCR re-runs text recognition over the whole document. It is more accurate on scans, takes noticeably longer, and can introduce recognition errors.',
    'reparse.cloud_warning': 'Cloud parsing uploads this document to a third-party service. The file leaves this machine and the call is billed.',

    'evidence.no_target.unknown_locator': 'This evidence reports its locator as unknown, so there is no visual targeting available for it.',
    'evidence.no_target.relative_bbox': 'This evidence carries a relative bounding box. Mapped onto the page it would highlight the wrong region, so no highlight box is offered.',
    'evidence.no_target.missing_bbox': 'This evidence has a page number but no bounding box, so it cannot be located precisely on the page.',
    'evidence.no_target.missing_page': 'This evidence has no page number, so it cannot be located on a page.',
    'evidence.no_target.malformed_bbox': 'This evidence’s bounding box coordinates are in the wrong order (far edge before near edge). Mapped onto the page it would mirror the wrong way, so no highlight box is offered.',
    'evidence.no_target.incomplete_locator': 'This evidence’s locator is incomplete, so there is no visual targeting available for it.',
    'evidence.no_target.fallback': 'No visual targeting is available for this evidence.',
    'evidence.capability.pdf_page_highlight': 'Can highlight on the page',
    'evidence.capability.slide_jump': 'Can jump to the slide',
    'evidence.capability.sheet_cells': 'Can locate the table region',
    'evidence.capability.element_jump': 'Can jump to the paragraph',
    'evidence.capability.offset_jump': 'Can jump to the character range',
    'evidence.view_kind.page_map': 'Page layout (rebuilt from parse coordinates)',
    'evidence.view_kind.slide_view': 'Slide content',
    'evidence.view_kind.element_view': 'Paragraph location',
    'evidence.view_kind.fallback': 'Located view',
    'evidence.view_target_missing': 'This evidence points at this page or paragraph, but the document’s parse result has no matching element, so its position cannot be marked in the view.',
    'evidence.highlight_drawn': 'The highlight box comes from this evidence’s own bounding box (absolute coordinates), using the same PDF user-space coordinates as the elements inside the page.',
    'evidence.highlight_absent': 'No highlight box drawn: the positions below cannot be given provably correct coordinates, so none is offered. The passage is still shown underneath.',
    'evidence.not_a_raster': 'Not a rendered image',
    'evidence.unsupported_view': 'This kind of location has no view to show.',

    'recommend.signal.cited_by': 'Cited by',
    'recommend.signal.citing': 'Citing',
    'recommend.signal.related': 'Related',
    'recommend.signal.topical': 'Topically similar',
    'recommend.signal.same_dataset': 'Same dataset',
    'recommend.signal.fallback': 'Signal',
    'recommend.state.ok.label': 'Fetched',
    'recommend.state.ok.summary': 'The source was queried and this signal returned matching papers.',
    'recommend.state.empty.label': 'Queried · no results',
    'recommend.state.empty.summary': 'The source was queried and genuinely has no matching papers. This is a conclusion about the literature itself.',
    'recommend.state.unavailable.label': 'Not fetched',
    'recommend.state.unavailable.summary': 'This signal’s result could not be fetched. That does not mean there are no related papers — there may be no literature identifier, an upstream rate limit, or a network error.',
    'recommend.state.skipped.label': 'Skipped',
    'recommend.state.skipped.summary': 'This request’s upstream call budget was spent, so this signal was not queried.',
    'recommend.confidence.declared': 'Name is declared',
    'recommend.confidence.derived': 'Name extracted from a paraphrase, unconfirmed',
    'recommend.confidence.unknown': 'Confidence not recorded',
    'recommend.reasons_heading': 'Why this was recommended',
    'recommend.no_reasons': 'This recommendation came with no signal reasons.',
    'recommend.duplicate.heading': 'Possible duplicates (not merged automatically)',
    'recommend.duplicate.policy': 'These two results may be different records of the same paper. They are unconfirmed, so both are kept and neither is merged. Merging two different papers would permanently lose one of them, while showing a duplicate twice only costs a second look.',
    'recommend.duplicate.badge': 'Possible duplicate',
    'recommend.duplicate.basis_fallback': 'Identical enough to be suspicious, but not yet confirmed to be the same paper',
    'recommend.duplicate.untitled': '(untitled result)',
    'recommend.signals_heading': 'Recommendation signal status',
    'recommend.paper_count': '{count} papers',

    'export.citations_heading': 'Citations and evidence links',
    'export.provenance_heading': 'How this was generated',
    'export.unknown_value': '(not recorded)',
    'export.presentation.direct_quote': 'Direct quote (verified verbatim against the source)',
    'export.presentation.paraphrase': 'Paraphrase (not verified verbatim, no quotation marks)',
    'export.fact.research_question': 'Research question',
    'export.fact.contributions': 'Main contributions',
    'export.fact.method': 'Method',
    'export.fact.datasets': 'Datasets',
    'export.fact.metrics': 'Evaluation metrics',
    'export.fact.main_results': 'Main results',
    'export.fact.limitations': 'Limitations',
    'export.fact.reproducibility': 'Reproducibility',
    'export.fact.unnamed': 'Unnamed field',
    'export.status.extracted': 'Extracted and backed by a citation',
    'export.status.needs_review': 'Did not pass evidence verification, needs manual review',
    'export.status.not_found': 'The paper does not state it',

    // --- the unrecognised code ---------------------------------------------
    // Shown when the server sends a `code` this build has no entry for. Saying
    // so is more useful than either inventing a sentence or passing off the
    // server's Chinese prose as if it were the interface's own.
    'error.unknown.title': 'The server returned an error code this version does not know',
    'error.unknown.message': 'The error code below came from the server, and this version has no description prepared for it. Repeating the same request will not change the result.',
    'error.unknown.hint': 'Please report this error code along with what you were doing to operations.',

    'switcher.group_label': 'Language / 语言',

    'skip.to_upload': 'Skip to upload',
    'skip.to_search': 'Skip to paper search',
    'skip.to_result': 'Skip to analysis workspace',
    'brand.subtitle': 'Research Workspace',
    'hero.copy': 'Upload a paper, extract structure, generate a map, and keep asking follow-up questions in the same workspace. This page now behaves more like a research desk than a single-use form.',
    'hero.stats.types': 'document types',
    'hero.stats.streaming': 'section streaming',
    'hero.stats.followups': 'follow-ups',
    'nav.upload': 'Upload',
    'nav.paper_search': 'Paper Search',
    'nav.reading_queue': 'Reading Queue',
    'nav.analysis': 'Analysis Workspace',
    'status.waiting': 'Waiting for document',

    'upload.workspace_title': 'Document upload workspace',
    'upload.intro': 'Supports `PDF`, `TXT`, `DOCX`, and `PPTX` files. A successful analysis replaces the current session; failed attempts keep the last completed workspace available.',
    'upload.shortcut_hints': 'Workspace keyboard shortcuts',
    'upload.api_key_label': 'API Key (Optional)',
    'upload.api_key_placeholder': 'Enter your API Key',
    'upload.source_label': 'Document Source',
    'upload.dropzone_label': 'Choose or drop a document file',
    'upload.dropzone_title': 'Drop a paper here or browse',
    'upload.dropzone_hint': 'PDF, TXT, DOCX, or PPTX · up to 16 MB',
    'upload.file_meta_empty': 'No file selected. Recommended: clean PDF, TXT, DOCX, or PPTX for better structure extraction.',
    'upload.opt_mermaid': 'Enable structure diagram',
    'upload.opt_evaluation': 'Enable critical evaluation',
    'upload.opt_brief': 'Enable research brief',
    'upload.analyze': 'Analyze Document',
    'upload.cancel_analysis': 'Cancel analysis',
    'upload.preparing': 'Preparing document analysis...',
    'progress.upload': 'Upload',
    'progress.analyze': 'Analyze',
    'progress.render': 'Render',
    'progress.ready': 'Ready',
    'progress.label': 'Analysis progress',

    'search.title': 'Paper Search',
    'search.query_label': 'Paper search query',
    'search.placeholder': 'Search by topic, task, method, or dataset...',
    'search.submit': 'Search Papers',
    'search.cancel': 'Cancel search',
    'search.meta': 'Search across Semantic Scholar and arXiv with a single query.',
    'search.examples_label': 'Example paper search queries',
    'queue.title': 'Reading Queue',
    'queue.clear': 'Clear',
    'queue.meta': 'Save papers from search or recommendations into this session reading queue.',
    'parse.title': 'Parse Status',
    'parse.reparse': 'Reparse',
    'parse.unknown_notice': 'The server did not report how this document was parsed.',
    'parse.cloud_result': 'Cloud parse outcome',
    'parse.cloud_outcome.success': 'Cloud parsing finished and returned text.',
    'parse.cloud_outcome.partial': 'Cloud parsing returned only part of the document, so this one may be incomplete.',
    'parse.cloud_outcome.quota_exhausted': 'The cloud parsing quota is spent, so this run got no cloud text.',
    'parse.cloud_outcome.timed_out': 'Cloud parsing stopped waiting before a result arrived.',
    'parse.cloud_outcome.blocked': 'Policy refused to hand this document to cloud parsing.',
    'parse.cloud_outcome.failed': 'Cloud parsing did not return a usable result.',
    'parse.cloud_outcome.unknown': 'The server reported a cloud outcome this version does not know.',
    'parse.fieldset_legend': 'Reparse this document',
    'parse.parser_label': 'Parser',
    'parse.force_ocr': 'Re-parse (this parser already OCRs by default)',
    'parse.submit': 'Reparse now',
    'export.title': 'Session Export',
    'export.format_label': 'Format',
    'export.submit': 'Export Session',
    'export.hint': 'Exports carry each citation’s evidence link, the document version and the analysis metadata (model, prompt version, evidence version, cost), so a file opened months later still states what produced it.',
    'export.preview_empty': 'Export the current analysis, Q&A history, research trace, and Mermaid assets after a successful run.',
    'result.overview': 'Overview',
    'result.citations': 'Key Citations',
    'result.structure': 'Text Structure',
    'result.visual_map': 'Visual Map',
    'result.evaluation': 'Evaluation',
    'result.brief': 'Deep Research Brief',
    'result.recommendations': 'Auto Recommendations',
    'result.recommend_submit': 'Recommend from current paper',
    'result.recommend_meta': 'Analyze a paper first, then generate follow-up reading suggestions from the current session.',
    'result.signals_label': 'Recommendation signal status',
    'result.ask': 'Ask Questions',
    'result.ask_label': 'Question about the current document',
    'result.ask_placeholder': 'Ask about evidence, methods, limitations, or reproducibility...',
    'result.ask_send': 'Send',
    'result.ask_cancel': 'Cancel',
    'result.answer_mode_label': 'Answer mode',
    'result.starter_label': 'Starter questions',
    'result.smart_prompts_label': 'Suggested follow-up questions',
    'result.next_actions_label': 'Recommended next actions',
    'result.copy': 'Copy',
    'result.recommend_cancel': 'Cancel',

    'footer.ai_label': 'AI collaborators',
    'footer.crafted_with': 'Crafted with',
    'evidence.heading': 'Evidence',
    'evidence.close': 'Close',
    'evidence.close_label': 'Close evidence panel',
    'evidence.idle_title': 'No citation open.',
    'evidence.idle_detail': 'Choose a citation chip to see the passage behind a claim. Your question and place in the conversation are kept.',
    'evidence.loading_title': 'Opening evidence…',
    'evidence.error_title': 'The evidence could not be opened.',
    'evidence.no_target_title': 'No visual targeting is available for this evidence.',
    'evidence.no_target_detail': 'The passage is still shown below.',
    'evidence.empty_title': 'This evidence resolved, but carries no text.',
    'evidence.empty_detail': 'The citation is real and points at an empty passage in the stored document.',
    'evidence.truncated': 'snippet truncated to the first 8000 characters',
    'evidence.characters': 'characters {start}–{end}',
    'evidence.page_images_unavailable': 'Page images are not rendered in this build.',
    'evidence.table_shell_empty': 'The parser produced an empty table shell; the cells below have no recovered content.',
    'evidence.reason_code': 'Reason code: {code}',
    'evidence.still_shown': 'The passage is still shown below.',
    'evidence.no_passage': 'This citation carries no passage text either.',
    'evidence.opening': 'Opening evidence...',
    'evidence.opened': 'Evidence opened',
    'evidence.closed': 'Evidence closed',
    'evidence.unavailable': 'Evidence unavailable',
    'evidence.failed': 'Could not open this evidence.',
    'evidence.no_document_id': 'This session has no document id, so its evidence cannot be addressed.',
    'evidence.id_not_published': 'The citation is real; the server has not published the document id it belongs to.',
    'evidence.citations_label': 'Citations',

    'claim.prefix.direct_quote': 'Verbatim quote:',
    'claim.prefix.normalized_quote': 'Aligned verbatim after normalisation:',
    'claim.prefix.paraphrase': 'Paraphrase (not verified verbatim against the source):',
    'claim.chip.direct_quote': 'Source-verified direct quote{label}, open the evidence',
    'claim.chip.paraphrase': 'A paraphrase that did not pass source verification, with no citation location',
    'claim.chips_group_label': 'Open evidence',
    'claim.chip_generic_label': 'Evidence',

    'evidence.page_label': 'Page {label}',
    'evidence.snippet.citation': 'Citation {label}',
    'evidence.snippet.verbatim': 'From the verified source text',
    'evidence.snippet.truncated': 'passage truncated',

    'parse.field.parser': 'Parser',
    'parse.field.media_type': 'Media type',
    'parse.field.status': 'Parse status',
    'parse.field.cache': 'Cache',
    'parse.value.not_reported': 'not reported',
    'parse.value.cache_hit': 'Served from cache; no parser ran this time.',
    'parse.notice.used_cloud': 'This document was uploaded to a cloud parsing service. Its contents left this machine.',
    'parse.notice.cloud_discarded': 'A cloud parser was called and charged, but its result was discarded; the text you see was parsed locally.',
    'parse.notice.local_only': 'No parser reached the network. This document stayed on this machine.',
    'parse.notice.fallback': 'The first parser did not produce a usable result; a fallback parser produced this one.',
    'parse.notice.failed': 'The parse failed: {message}',
    'parse.quality.absent': 'No parse quality report was returned for this document.',
    'parse.quality.heading': 'Parse quality: {status}',
    'parse.quality.clean': 'Every checked rule passed.',
    'parse.quality.rule_fallback': 'rule',

    'reparse.parser.default': 'The deployment’s configured route (chosen per file type)',
    'reparse.parser.docling': 'Docling (local)',
    'reparse.parser.docling_ocr': 'Docling with forced OCR (local)',
    'reparse.parser.mineru_cloud': 'MinerU cloud parser',
    'reparse.parser.default_note': 'This entry names no parser: the deployment picks a route per file type, and different file types can take different parsers.',
    'reparse.parser.requires_gateway': 'Needs an administrator to set PARSER_BACKEND=gateway; otherwise the server refuses with 409 parser_unavailable.',
    'reparse.parser.requires_gateway_ocr': 'Needs an administrator to set PARSER_BACKEND=gateway; otherwise the server refuses with 409 parser_unavailable. The forced OCR is run by the chosen Docling itself.',
    'reparse.parser.requires_cloud': 'Needs a cloud parsing adapter in this deployment, a privacy policy that allows egress, and remaining call quota.',
    'reparse.hint.ready': 'Reparse this document with the selected options.',
    'reparse.hint.no_document': 'No document id is available for this session, so there is nothing to reparse.',
    'reparse.result.started': 'Reparsing...',
    'reparse.result.finished': 'Reparse finished. The parse status above reflects the new run.',
    'reparse.result.no_document': 'This session has no document id, so there is nothing to reparse.',
    'reparse.result.no_route': 'This server does not offer a reparse endpoint (POST {path} → {reason}). The document was left untouched.',
    'reparse.result.failed': 'The reparse request did not succeed ({reason}). The document was left untouched.',
    'reparse.status.started': 'Reparsing document...',
    'reparse.status.finished': 'Reparse finished',
    'reparse.status.unavailable': 'Reparse unavailable',

    'a11y.shortcut_upload': 'Upload shortcut focused document source',
    'a11y.shortcut_search': 'Search shortcut focused paper search',
    'a11y.shortcut_ask': 'Ask shortcut focused question input',
    'theme.switch_to_light': 'Switch to light theme',
    'theme.switch_to_dark': 'Switch to dark theme',
    'mermaid.failed_to_load': 'Mermaid failed to load.',
    'mermaid.too_complex': 'Structure too complex to render. Raw data fallback:',
    'content.none_available': 'No content available.',

    'error.request_failed': 'Request failed.',
    'footer.hide_ai': 'Hide AI collaborators',
    'footer.show_ai': 'Show AI collaborators',

    'copy.no_content': 'No content',
    'copy.nothing_yet': 'Nothing to copy yet',
    'copy.done': 'Copied',
    'copy.done_status': 'Content copied',
    'copy.failed': 'Copy failed',
    'copy.failed_status': 'Copy failed'
};

/**
 * Every localisable string, in every shipped language.
 *
 * Flat, dotted keys rather than nested objects: a lookup is one property read,
 * a missing key is obvious in a test rather than a deep `undefined`, and the
 * `tests/test_i18n.py` key-parity assertion is a set comparison.
 */
export const CATALOGUE = { 'zh-CN': ZH_CN, en: EN };

// ==========================================================================
//
// The lookup
// ==========================================================================

/**
 * Resolve a key in the active locale.
 *
 * Falls back to {@link DEFAULT_LOCALE} and then, as a last resort, to the key
 * itself. Returning the key is deliberate: a missing translation should be
 * visible in the interface as `export.fact.method` during review, not as a
 * silently empty button.
 *
 * @param {string} key       a dotted catalogue key
 * @param {object} [params]  `{name}` placeholders, replaced once each
 */
export function t(key, params) {
    const table = CATALOGUE[currentLocale] || CATALOGUE[DEFAULT_LOCALE];
    const fallback = CATALOGUE[DEFAULT_LOCALE];
    const value = (table && table[key] !== undefined) ? table[key] : fallback[key];
    if (value === undefined) return key;
    return interpolate(String(value), params);
}

/** The active locale. */
export function getLocale() {
    return currentLocale;
}

function interpolate(template, params) {
    if (!params) return template;
    return template.replace(/\{(\w+)\}/g, (match, name) => (
        Object.prototype.hasOwnProperty.call(params, name) && params[name] !== undefined && params[name] !== null
            ? String(params[name])
            : match
    ));
}

// ==========================================================================
//
// Numbers, dates and units
// ==========================================================================

/**
 * A count, grouped and digit-shaped for the active locale.
 *
 * A hard-coded "4 pages" is a localisation bug: `en` groups thousands and
 * renders `1,024`, `zh-CN` does the same but formats decimals and dates
 * differently, and neither is negotiable. Everything numeric a reader sees goes
 * through here rather than through `String(n)`.
 */
export function formatNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value ?? '');
    try {
        return new Intl.NumberFormat(currentLocale).format(number);
    } catch {
        return String(number);
    }
}

/**
 * A timestamp, in the active locale's date and time convention.
 *
 * Both the export filename stem and the on-screen provenance are affected:
 * `2026/09/30 17:33` and `30/09/2026 17:33` are the same instant and the
 * reader has to recognise their own.
 */
export function formatDateTime(value) {
    const date = value instanceof Date ? value : new Date(value);
    if (Number.isNaN(date.getTime())) return String(value ?? '');
    try {
        return new Intl.DateTimeFormat(currentLocale, {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        }).format(date);
    } catch {
        return date.toISOString();
    }
}

/** A byte count, in the active locale's units. */
export function formatFileSize(bytes) {
    const number = Number(bytes);
    if (!Number.isFinite(number) || number < 0) return String(bytes ?? '');
    const units = ['B', 'KB', 'MB', 'GB'];
    let index = 0;
    let value = number;
    while (value >= 1024 && index < units.length - 1) {
        value /= 1024;
        index += 1;
    }
    const digits = (value >= 10 || index === 0) ? 0 : 1;
    try {
        return `${new Intl.NumberFormat(currentLocale, {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits
        }).format(value)} ${units[index]}`;
    } catch {
        return `${value.toFixed(digits)} ${units[index]}`;
    }
}

// ==========================================================================
//
// Errors: driven by `code`, never by the server's prose
// ==========================================================================

/**
 * The reader-facing copy for every backend `code`, in both languages.
 *
 * **This is the answer to the mixed-language problem.** The backend writes its
 * `error` field in Chinese, always, and that text has no `lang` attribute and
 * no declared language — so surfacing it under an English interface produces
 * an English page with a Chinese sentence stranded in it. The `code` beside it
 * *is* stable and language-neutral, so every message here is keyed by `code`
 * and authored per language. The server's prose is used only for a code this
 * build has never heard of, where showing the operator's raw code and the
 * server's own words is more useful than inventing a sentence.
 *
 * The shape mirrors the `{kind, title, message, hint}` the runtime already
 * consumes: `kind` stays in `config.js` (it is policy, not copy), and only the
 * three strings live here.
 */
const ERROR_COPY = {
    'zh-CN': {
        analysis_result_not_local: ['分析结果在另一个进程里', '分析已经在服务端完成，但结果不在本页所在的进程里，因此这里拿不到它。', '请用上面提示的任务编号打开对应的会话；重新提交同一份文件只会再解析一次，不会把这边的结果取回来。'],
        analysis_wait_timeout: ['等待分析超时', '服务端仍在处理这份文档，这次连接等到了等待上限就先返回了。', '分析本身没有被取消，也不需要重跑；请用上面提示的任务编号继续查看它的进度。'],
        storage_unavailable: ['存储暂时不可用', '服务端的数据存储现在无法访问。这次请求没有被处理，稍后重试即可。', ''],
        job_store_unavailable: ['任务存储暂时不可用', '服务端无法记录任务状态。这是一次服务故障，不是你的请求有问题，稍后重试。', ''],
        search_failed: ['论文检索失败', '检索服务在处理这次检索时出错。可以重试；如果反复失败，请检查检索服务的配置。', ''],
        paper_search_failed: ['论文检索失败', '检索服务在处理这次检索时出错。可以重试；如果反复失败，请检查检索服务的配置。', ''],
        paper_recommendation_failed: ['生成推荐失败', '推荐服务在处理这次请求时出错。推荐列表已被清空，页面上没有留下不完整的结果冒充完整的推荐。', ''],
        paper_download_failed: ['无法下载论文文件', '远端站点没有把文件交出来（可能限流、拒绝访问，或暂时不可达）。可以重试。', ''],
        paper_import_failed: ['导入论文失败', '从远端地址取回这篇论文时出错。没有创建文档，可以重试。', ''],
        question_answering_failed: ['回答失败', '问答服务在生成这次回答时出错。已经流式输出的部分保留在对话里，重试会重新回答整题。', ''],
        streaming_question_failed: ['回答流中断', '回答在流式输出途中失败。已输出的内容保留在对话里，重试会重新回答整题。', ''],
        streaming_analysis_failed: ['分析流中断', '分析在流式输出途中失败。这份文档没有被标记为已分析，可以重新分析。', ''],
        streaming_upload_failed: ['上传失败', '文件在上传途中失败。你选择的文件和所有选项都还在，直接重试即可。', ''],
        document_upload_failed: ['上传文档失败', '服务端在接收文件时出错。没有创建文档，可以重试。', ''],
        document_analysis_failed: ['文档分析失败', '分析这份文档时服务端出错。之前成功分析出的工作区已经恢复，可以重试。', ''],
        reading_queue_save_failed: ['阅读队列未能保存', '队列已经在页面上更新，但没能写入服务端会话。刷新页面会丢失这次改动。', '队列仍然保留在当前页面上，但尚未持久化。'],
        export_render_failed: ['导出渲染失败', '服务端在生成导出文件时出错。导出格式的选择仍然保留，可以重试。', ''],
        reparse_failed: ['重新解析失败', '重新解析没有完成。文档保持原样，没有被改动；你选的解析器和 OCR 选项仍然保留。', ''],
        document_busy: ['文档已有任务在进行', '这份文档还有一个未结束的任务。等它结束（或取消它）之后再提交。', ''],
        parser_timeout: ['解析器超时', '解析这份文档超过了服务端允许的时间上限。可以重试，或换一条更快的解析路径。', ''],
        parser_unavailable: ['所选解析器不可用', '当前部署没有启用你选择的解析器。改选另一个解析器，或联系管理员启用它。', ''],
        cloud_adapter_unavailable: ['云端解析未配置', '这个部署没有可用的云端解析适配器，因此云端回退没有发生，文档仍由本地解析处理。', ''],
        sandbox_failed: ['解析沙箱出错', '解析进程的隔离环境出错。可以重试。', ''],
        sandbox_unavailable: ['解析沙箱不可用', '服务端没有可用的解析隔离环境。这一条需要管理员处理。', ''],
        rlimit_in_serving_process: ['服务端资源限制已触发', '服务端在处理这次请求时触发了资源上限，因此中途放弃。可以稍后重试。', ''],
        privacy_gate_error: ['隐私策略无法判定', '服务端在检查这份文档是否可以出网时出错，出于谨慎它拒绝了云端解析。请联系管理员确认隐私策略配置。', ''],
        missing_file: ['没有选择文件', '请先选择一个文档。', ''],
        missing_filename: ['文件没有名称', '这个文件没有可用的文件名，服务端无法为它登记来源。', ''],
        missing_question: ['问题是空的', '请先输入一个问题再发送。', ''],
        missing_search_query: ['检索关键词是空的', '请先输入检索关键词。', ''],
        missing_paper_url: ['缺少论文下载地址', '这条结果没有可下载的文件地址，因此无法导入。', ''],
        missing_seed_paper: ['缺少论文标识', '推荐至少需要一个可检索的论文标识。这份文档只有标题，没有 DOI 或 arXiv id。', ''],
        missing_session_id: ['缺少会话标识', '这次请求没有带上会话标识，服务端无法知道它属于哪个会话。', ''],
        invalid_upload_metadata: ['文件信息不完整', '服务端无法读取这个文件的元数据（文件名或大小）。请换一个文件。', ''],
        invalid_document_payload: ['解析结果格式不合法', '提交的解析结果不符合服务端要求的结构。这是接口内部的不一致，请把这次操作报告给运维。', ''],
        unsupported_file_type: ['不支持的文件类型', '服务端不接受这个文件类型。请改用受支持的格式。', ''],
        unsupported_export_format: ['不支持的导出格式', '这个导出格式在当前部署中没有实现。导出没有开始，你选的格式仍然保留。', ''],
        upload_rejected: ['文件未通过校验', '服务端拒绝了这个文件。文件没有被保存，可以重试；反复失败请换一个文件。', ''],
        bad_request: ['请求被拒绝', '服务端拒绝了这次请求，但没有给出更具体的原因。', ''],
        invalid_request: ['请求内容不合法', '服务端无法接受这次请求的内容。没有产生任何改动。', ''],
        invalid_json: ['请求体不是合法 JSON', '这次请求的 JSON 无法解析。这是一个客户端问题，请把这次操作报告给运维。', ''],
        json_not_object: ['请求体不是 JSON 对象', '服务端要求这次请求的 JSON 是一个对象，收到的是别的。', ''],
        body_read_failed: ['服务端无法读取请求体', '服务端在读取这次请求的内容时出错。没有产生任何改动。', ''],
        malformed_id: ['标识格式不合法', '这次请求里的文档 id 或证据 id 格式不对，因此没有发起查询。', ''],
        invalid_page: ['页码不合法', '页码必须是从 1 开始的整数。', ''],
        page_out_of_range: ['页码超出范围', '这条证据指向的页码不在文档的页数范围内。这可能是页码映射的错误，而不是你操作错了。', ''],
        invalid_since: ['事件序号不合法', 'since_sequence 必须是不小于 0 的整数。', ''],
        invalid_state_filter: ['未知的任务状态', '这个过滤值不是服务端认识的任务状态。', ''],
        unknown_parser: ['未知的解析器', '选择的解析器名字不在服务端支持的列表里。重新解析没有被提交，你的选择仍然保留。', ''],
        missing_api_key: ['缺少 API Key', '这次调用 AI 服务需要一个 API Key，但它没有随请求提供。请在上方填入 API Key 后重试。', '你填入的 API Key 不会被保存，也不会随导出一起离开这台机器。'],
        missing_credential: ['缺少访问凭据', '这次请求没有携带凭据，因此被拒绝。', ''],
        invalid_credential: ['凭据无效或已失效', '这次请求携带的凭据没有通过校验。请检查 API Key 是否正确、是否已过期或已被撤销。', ''],
        principal_not_bound: ['凭据没有绑定到身份', '凭据通过了校验，但没有绑定到任何身份，因此被拒绝。', ''],
        insufficient_scope: ['权限范围不足', '这次凭据没有执行该操作所需的权限范围。', ''],
        cross_tenant_access_denied: ['跨会话访问被拒绝', '这次请求试图访问不属于该凭据的会话，因此被拒绝。', ''],
        session_not_found: ['会话不存在或已过期', '这个会话已经不在服务端了。重新分析一份文档即可开始新的会话。', ''],
        session_forbidden: ['会话令牌无效', '这次请求的会话令牌没有通过校验。重新分析一份文档会得到一个新的令牌。', ''],
        session_token_invalid: ['会话令牌无效', '这次请求的会话令牌没有通过校验。重新分析一份文档会得到一个新的令牌。', ''],
        invalid_session_token: ['会话令牌无效', '这次请求的会话令牌没有通过校验。重新分析一份文档会得到一个新的令牌。', ''],
        document_not_found: ['文档不存在', '服务端没有这个文档 id。它可能已经被删除，或者这个 id 来自另一次会话。', ''],
        evidence_not_found: ['证据不存在', '服务端没有这条证据。它可能属于另一份文档，或者已经被删除。', ''],
        evidence_not_in_document: ['证据不属于这份文档', '这条证据的 id 是有效的，但不属于你正在看的这份文档。', ''],
        document_exists: ['同名文档已存在', '服务端已经有同名的文档。重新上传不会覆盖它。', ''],
        original_unavailable: ['原始文件不在存储中', '这份文档的原始文件已经不在对象存储里，因此无法重新解析。需要重新上传同一份文件。', ''],
        job_not_found: ['任务不存在', '服务端没有这个任务。它可能已经超过保留期。', ''],
        job_not_retryable: ['这个任务不能重试', '只有处于“失败”状态的任务才能重新入队。这个任务不是失败状态，所以它没有被重新排队，也没有重跑。', ''],
        job_cancelled: ['任务已取消', '这个任务被取消了，没有产出文档。文档未被修改。', ''],
        document_parse_failed: ['文档解析失败', '这份文档没有解析出来，因此没有生成文档。文档未被修改。', '如果是扫描件，可以试“强制 OCR”重新解析；也可以换一个解析器。'],
        unreadable: ['文件里没有可读文字', '这个文件里没有可读取的正文（可能是纯图片或已经损坏）。请换一个有文字层的文件。', ''],
        too_many_pages: ['页数超出上限', '这份文档的页数超过了服务端允许的上限。同一份文件再传一次仍然是同一个结果。', '请只上传需要分析的部分。'],
        input_too_large: ['输入超出上限', '这份文件超过了服务端允许的大小上限。同一份文件再传一次仍然是同一个结果。', ''],
        output_too_large: ['解析结果超出上限', '这份文档解析后的内容量超过了服务端允许的上限，因此被拒绝，以免产生不可用的结果。', '请把文档拆小，或者只分析其中一部分。'],
        memory_limit_exceeded: ['超出内存上限', '解析这份文档需要的内存超过了服务端上限。同一份文件再传一次仍然是同一个结果。', '请把文档拆小，或者只分析其中一部分。'],
        cpu_limit_exceeded: ['超出 CPU 时间上限', '解析这份文档需要的计算时间超过了服务端上限。同一份文件再传一次仍然是同一个结果。', ''],
        sandbox_unpicklable: ['解析结果无法传回主进程', '解析进程产生的结果无法安全地传回，因此被拒绝。这是实现层的问题，请报告给运维。', ''],
        mineru_quota_exhausted: ['云端解析额度已用尽', '云端解析服务的调用额度已经用尽。立即重试仍然是同一个结果。', '把解析器改回本地（Docling）即可继续分析。'],
        cloud_parse_timed_out: ['云端解析在拿到结果前停止了等待', '云端解析在等待结果时到达了时间上限，因此这一次没有拿到云端文本。这个任务可能仍在服务商那边进行。', '重试是值得的：这份工作可能已经在云端完成，再解析一次就是再取一次结果。'],
        cloud_parse_blocked: ['云端解析被策略拒绝', '这个部署的隐私策略拒绝把这份文档交给云端解析，因此没有发起任何云端请求，文档也没有离开这台机器。', '重复同一个请求仍然会被同样拒绝；这需要管理员调整文档隐私策略。'],
        cloud_parse_failed: ['云端解析失败', '云端解析没有返回可用的结果，而且原因没有被进一步分类。下面这份文本来自另一条解析路径。', '服务端没有把更具体的原因随结果一起送上来，因此这里不猜它是什么。'],
        mineru_no_attempt: ['云端解析没有真正执行', '这次请求没有向云端解析发起任何调用，所以重试不可能改变结果。请检查服务端配置。', ''],
        mineru_ssrf_blocked: ['结果地址被安全策略拦截', '云端解析给出的结果地址指向内网或非法地址，出于安全考虑被拒绝。这是安全策略的结果，不是网络故障。', ''],
        mineru_zip_bomb: ['云端结果压缩包异常', '云端解析返回的压缩包解压比异常，出于安全考虑被拒绝。', ''],
        mineru_bad_zip: ['云端结果不是有效压缩包', '云端解析返回的内容不是一个可读的压缩包。同一份文件再传一次不会改变结果。', ''],
        mineru_bad_json: ['云端结果不是合法 JSON', '云端解析返回的索引文件无法解析为 JSON。', ''],
        mineru_bad_json_shape: ['云端结果结构不符', '云端解析返回的 JSON 结构与约定不符。', ''],
        mineru_bad_code: ['云端解析返回异常状态', '云端解析服务返回的状态码不在约定范围内。', ''],
        mineru_bad_content_list: ['云端结果缺少内容列表', '云端解析的返回里没有内容列表，因此没有可用的解析结果。', ''],
        mineru_empty_result: ['云端解析没有产出内容', '云端解析完成了，但没有产出任何内容。这份文档在这条路径上没有结果。', ''],
        mineru_incomplete_batch: ['云端解析批次不完整', '云端解析只返回了批次的一部分，其余页面没有被解析，因此整体结果不可用。', ''],
        mineru_local_read_error: ['无法读取云端结果文件', '结果已经下载，但本地读取失败。请检查服务端的磁盘权限。', ''],
        mineru_response_too_large: ['云端解析响应超出上限', '云端解析服务返回的内容超过了服务端允许的大小上限，因此被拒绝。', ''],
        mineru_missing_result_url: ['云端解析没有给出结果地址', '云端解析服务没有返回结果文件的位置，因此没有拿到结果。', ''],
        mineru_empty_response: ['云端解析返回为空', '云端解析服务返回了空响应。可以重试。', ''],
        mineru_network_error: ['云端解析网络错误', '调用云端解析服务时网络不通。可以重试。', ''],
        mineru_http_error: ['云端解析服务返回错误', '云端解析服务返回了一个错误状态。可以稍后重试。', ''],
        mineru_business_error: ['云端解析业务错误', '云端解析服务报告了一次业务错误（文件不被接受，或任务参数有问题）。可以重试一次。', ''],
        mineru_task_failed: ['云端解析任务失败', '云端解析的任务执行失败。可以重试。', ''],
        archive_bomb: ['压缩包展开后过大', '这个压缩包解压后的体积异常，出于安全考虑被拒绝。', ''],
        archive_nested_bomb: ['嵌套压缩包展开后过大', '压缩包里的压缩包解压后体积异常，出于安全考虑被拒绝。', ''],
        archive_nested_too_deep: ['压缩包嵌套层数过深', '这个压缩包的嵌套层数超过了服务端允许的上限。', ''],
        archive_traversal: ['压缩包包含越界路径', '压缩包里的路径试图写出解压目录，出于安全考虑被拒绝。', ''],
        archive_symlink: ['压缩包包含符号链接', '压缩包里包含指向解压目录之外的链接，出于安全考虑被拒绝。', ''],
        consent_not_granted: ['云端解析未获授权', '隐私策略允许云端解析，但还没有获得你的显式同意。文档没有被上传，解析没有开始。', '勾选云端解析的同意项之后再试。'],
        privacy_mode_local_only: ['云端解析被隐私策略禁止', '当前文档隐私模式禁止任何文档出网。文档没有被上传，解析没有开始。', '这需要管理员把 DOCUMENT_PRIVACY_MODE 改为 allow_cloud。'],
        page_render_unavailable: ['这个版本没有页图渲染能力', '当前部署没有安装可选的渲染依赖，因此无法给出这一页的原始渲染图。证据面板改用解析得到的元素坐标重建页内布局，并会明确标注它不是原始渲染图。', '安装可选的渲染依赖后，这个入口会开始返回页图。'],
        page_render_unsupported_format: ['页图渲染不支持这种文件', '渲染器只能处理它认识的文档格式，这份文档不在其中。', ''],
        page_render_failed: ['页图渲染失败', '渲染这一页时出错，因此没有产出图像。原文片段仍然可以查看。', ''],
        client_timeout: ['请求超时', '在等待响应的时间上限内没有收到任何答复，请求已被中止。服务端可能仍在处理它。', '长任务不会因为这里超时而停止，可以重新连接查看它到了哪一步。'],
        client_unreachable: ['连不上服务端', '这个地址没有响应：可能是网络断了、服务没在运行，或者地址指向了别的地方。', ''],
        client_offline: ['当前处于离线状态', '浏览器报告这台设备没有网络连接。已排队的操作会在恢复连接后发出。', ''],
        client_malformed_response: ['响应无法解析', '服务端返回的内容不是这个接口约定的格式。页面上其余部分不受影响。', '这通常意味着服务端的版本与这个页面不一致。'],
        client_response_too_large: ['响应超出大小上限', '服务端返回的内容超过了这个页面设定的读取上限，已在读取途中停止。', ''],
        client_stream_interrupted: ['连接中断', '数据流在传输途中断开。服务端可能已经完成了这次工作，重新连接可以确认，而不必重做。', ''],
        client_cancelled: ['已取消', '这次操作被取消了，没有产生结果。', ''],
        client_unknown_error: ['发生了一个没有分类的错误', '这次操作失败了，但失败的原因既不来自服务端响应，也不来自网络。页面上没有留下半成品。', '请把当时的操作步骤告诉运维。']
    },
    en: {
        analysis_result_not_local: ['The analysis result is in another process', 'The analysis finished on the server, but the result is not in the process serving this page, so it cannot be handed back here.', 'Open the session named by the job id above; submitting the same file again would only parse it a second time and would not bring this result back.'],
        analysis_wait_timeout: ['Timed out waiting for the analysis', 'The server is still working on this document; this connection returned when its wait limit was reached.', 'The analysis itself was not cancelled and does not need to be run again — follow the job id above to watch its progress.'],
        storage_unavailable: ['Storage is temporarily unavailable', 'The server’s data storage cannot be reached. This request was not processed; retrying later is enough.', ''],
        job_store_unavailable: ['Job storage is temporarily unavailable', 'The server cannot record job state. This is a service fault, not a problem with your request; retry later.', ''],
        search_failed: ['Paper search failed', 'The search service hit an error handling this query. You can retry; if it keeps failing, check the search service configuration.', ''],
        paper_search_failed: ['Paper search failed', 'The search service hit an error handling this query. You can retry; if it keeps failing, check the search service configuration.', ''],
        paper_recommendation_failed: ['Generating recommendations failed', 'The recommendation service hit an error on this request. The recommendation list was cleared rather than leaving a partial result masquerading as a complete one.', ''],
        paper_download_failed: ['Could not download the paper file', 'The remote site did not hand over the file (it may be rate limiting, denying access, or temporarily unreachable). You can retry.', ''],
        paper_import_failed: ['Importing the paper failed', 'Something went wrong fetching this paper from its remote URL. No document was created; you can retry.', ''],
        question_answering_failed: ['Answering failed', 'The question-answering service hit an error generating this answer. Whatever streamed out already stays in the conversation, and a retry re-answers the whole question.', ''],
        streaming_question_failed: ['The answer stream was interrupted', 'The answer failed partway through streaming. What was already streamed stays in the conversation, and a retry re-answers the whole question.', ''],
        streaming_analysis_failed: ['The analysis stream was interrupted', 'The analysis failed partway through streaming. This document is not marked as analysed, so you can analyse it again.', ''],
        streaming_upload_failed: ['Upload failed', 'The file failed partway through upload. The file you chose and every option are still set, so you can retry directly.', ''],
        document_upload_failed: ['Uploading the document failed', 'The server hit an error receiving the file. No document was created; you can retry.', ''],
        document_analysis_failed: ['Document analysis failed', 'The server hit an error analysing this document. The last successfully analysed workspace has been restored, and you can retry.', ''],
        reading_queue_save_failed: ['The reading queue was not saved', 'The queue was updated on the page but not written to the server session. Reloading the page loses this change.', 'The queue is still on this page, but it is not persisted.'],
        export_render_failed: ['Export rendering failed', 'The server hit an error generating the export file. Your format choice is still set, and you can retry.', ''],
        reparse_failed: ['Reparsing failed', 'The reparse did not complete. The document is untouched; the parser and OCR options you chose are still set.', ''],
        document_busy: ['This document already has a job running', 'This document still has an unfinished job. Wait for it to finish (or cancel it) before submitting again.', ''],
        parser_timeout: ['The parser timed out', 'Parsing this document exceeded the server’s time limit. You can retry, or take a faster parsing path.', ''],
        parser_unavailable: ['The chosen parser is unavailable', 'This deployment has not enabled the parser you chose. Pick a different one, or ask an administrator to enable it.', ''],
        cloud_adapter_unavailable: ['Cloud parsing is not configured', 'This deployment has no cloud parsing adapter available, so the cloud fallback did not happen and the document was still handled by local parsing.', ''],
        sandbox_failed: ['The parsing sandbox failed', 'The parse process’s isolation environment failed. You can retry.', ''],
        sandbox_unavailable: ['The parsing sandbox is unavailable', 'The server has no parsing isolation environment available. This one needs an administrator.', ''],
        rlimit_in_serving_process: ['A server resource limit was hit', 'The server hit a resource ceiling on this request and gave up partway. You can retry later.', ''],
        privacy_gate_error: ['The privacy policy could not be decided', 'The server hit an error while checking whether this document may leave the machine, and refused cloud parsing out of caution. Ask an administrator to confirm the privacy policy configuration.', ''],
        missing_file: ['No file chosen', 'Choose a document first.', ''],
        missing_filename: ['The file has no name', 'This file has no usable filename, so the server cannot record where it came from.', ''],
        missing_question: ['The question is empty', 'Type a question before sending.', ''],
        missing_search_query: ['The search query is empty', 'Type search keywords first.', ''],
        missing_paper_url: ['No downloadable paper URL', 'This result has no downloadable file URL, so it cannot be imported.', ''],
        missing_seed_paper: ['No paper identity', 'A recommendation needs at least one searchable paper identifier. This document has only a title, with no DOI or arXiv id.', ''],
        missing_session_id: ['No session identifier', 'This request carried no session identifier, so the server cannot tell which session it belongs to.', ''],
        invalid_upload_metadata: ['The file’s metadata is incomplete', 'The server cannot read this file’s metadata (name or size). Please use a different file.', ''],
        invalid_document_payload: ['The parse result is malformed', 'The submitted parse result does not match the structure the server requires. This is an internal inconsistency; please report this operation to operations.', ''],
        unsupported_file_type: ['Unsupported file type', 'The server does not accept this file type. Please use one of the supported formats.', ''],
        unsupported_export_format: ['Unsupported export format', 'This export format is not implemented in this deployment. The export did not start, and your format choice is still set.', ''],
        upload_rejected: ['The file failed validation', 'The server refused this file. It was not saved, and you can retry; if it keeps failing, use a different file.', ''],
        bad_request: ['The request was refused', 'The server refused this request but gave no more specific reason.', ''],
        invalid_request: ['The request content is not valid', 'The server cannot accept the content of this request. Nothing was changed.', ''],
        invalid_json: ['The request body is not valid JSON', 'This request’s JSON could not be parsed. This is a client-side problem; please report this operation to operations.', ''],
        json_not_object: ['The request body is not a JSON object', 'The server requires this request’s JSON to be an object, and received something else.', ''],
        body_read_failed: ['The server could not read the request body', 'The server hit an error reading the content of this request. Nothing was changed.', ''],
        malformed_id: ['The identifier is malformed', 'The document id or evidence id in this request is not well formed, so no lookup was made.', ''],
        invalid_page: ['The page number is not valid', 'The page number must be an integer starting at 1.', ''],
        page_out_of_range: ['The page number is out of range', 'The page this evidence points at is not within the document’s page count. This may be a page-mapping error rather than a mistake on your part.', ''],
        invalid_since: ['The event sequence number is not valid', 'since_sequence must be an integer greater than or equal to 0.', ''],
        invalid_state_filter: ['Unknown job state', 'This filter value is not a job state the server recognises.', ''],
        unknown_parser: ['Unknown parser', 'The chosen parser name is not in the server’s supported list. The reparse was not submitted, and your choice is still set.', ''],
        missing_api_key: ['No API Key', 'Calling the AI service needs an API Key, and none came with this request. Enter an API Key above and try again.', 'The API Key you enter is not saved, and does not leave this machine with an export.'],
        missing_credential: ['No access credential', 'This request carried no credential, so it was refused.', ''],
        invalid_credential: ['The credential is invalid or has expired', 'The credential on this request did not pass validation. Check that the API Key is correct, not expired, and not revoked.', ''],
        principal_not_bound: ['The credential is not bound to an identity', 'The credential passed validation but is not bound to any identity, so it was refused.', ''],
        insufficient_scope: ['Insufficient scope', 'This credential does not have the scope needed to perform the operation.', ''],
        cross_tenant_access_denied: ['Cross-session access refused', 'This request tried to reach a session that does not belong to this credential, so it was refused.', ''],
        session_not_found: ['The session does not exist or has expired', 'This session is no longer on the server. Analysing a document again starts a new session.', ''],
        session_forbidden: ['The session token is not valid', 'This request’s session token did not pass validation. Analysing a document again gets you a new token.', ''],
        session_token_invalid: ['The session token is not valid', 'This request’s session token did not pass validation. Analysing a document again gets you a new token.', ''],
        invalid_session_token: ['The session token is not valid', 'This request’s session token did not pass validation. Analysing a document again gets you a new token.', ''],
        document_not_found: ['The document does not exist', 'The server has no such document id. It may have been deleted, or the id may come from another session.', ''],
        evidence_not_found: ['The evidence does not exist', 'The server has no such evidence. It may belong to another document, or it may have been deleted.', ''],
        evidence_not_in_document: ['The evidence is not in this document', 'This evidence id is valid, but it does not belong to the document you are looking at.', ''],
        document_exists: ['A document with that name already exists', 'The server already has a document with this name. Uploading again will not overwrite it.', ''],
        original_unavailable: ['The original file is not in storage', 'This document’s original file is no longer in object storage, so it cannot be reparsed. You need to upload the same file again.', ''],
        job_not_found: ['The job does not exist', 'The server has no such job. It may have passed its retention period.', ''],
        job_not_retryable: ['This job cannot be retried', 'Only a job in the “failed” state can be re-queued. This job is not in a failed state, so it was neither re-queued nor re-run.', ''],
        job_cancelled: ['The job was cancelled', 'This job was cancelled and produced no document. The document was not modified.', ''],
        document_parse_failed: ['Document parsing failed', 'This document did not parse, so no document was created. The document is unmodified.', 'If it is a scan, try reparsing with forced OCR, or try a different parser.'],
        unreadable: ['No readable text in the file', 'This file has no readable body text (it may be images only, or already corrupt). Please use a file with a text layer.', ''],
        too_many_pages: ['Page count over the limit', 'This document’s page count exceeds the server’s limit. Uploading the same file again gives the same result.', 'Please upload only the part you need analysed.'],
        input_too_large: ['Input over the limit', 'This file exceeds the server’s size limit. Uploading the same file again gives the same result.', ''],
        output_too_large: ['Parse result over the limit', 'This document’s parsed content volume exceeds the server’s limit and was refused, to avoid producing an unusable result.', 'Split the document, or analyse only part of it.'],
        memory_limit_exceeded: ['Over the memory limit', 'Parsing this document needs more memory than the server allows. Uploading the same file again gives the same result.', 'Split the document, or analyse only part of it.'],
        cpu_limit_exceeded: ['Over the CPU time limit', 'Parsing this document needs more compute time than the server allows. Uploading the same file again gives the same result.', ''],
        sandbox_unpicklable: ['The parse result could not be returned to the main process', 'The parse process produced a result that cannot be safely returned, so it was refused. This is an implementation problem; please report it to operations.', ''],
        mineru_quota_exhausted: ['Cloud parsing quota is exhausted', 'The cloud parsing service’s call quota is spent. Retrying immediately gives the same result.', 'Switch the parser back to local (Docling) to keep analysing.'],
        cloud_parse_timed_out: ['Cloud parsing stopped waiting for a result', 'Cloud parsing hit its wait limit before a result arrived, so this run got no cloud text. The job may still be running on the provider’s side.', 'Retrying is worth it: the work may already have finished on the cloud, and re-parsing asks for that result again.'],
        cloud_parse_blocked: ['Cloud parsing was refused by policy', 'This deployment’s privacy policy refused to hand the document to cloud parsing, so no cloud request was made and nothing left this machine.', 'Repeating the same request is refused the same way; an administrator has to change the document privacy policy.'],
        cloud_parse_failed: ['Cloud parsing failed', 'Cloud parsing did not return a usable result and the cause was not classified further. The text below came from a different parsing route.', 'The provider did not send a specific reason along with the result, so this does not guess at one.'],
        mineru_no_attempt: ['Cloud parsing never actually ran', 'This request made no call to cloud parsing, so a retry cannot change the result. Please check the server configuration.', ''],
        mineru_ssrf_blocked: ['The result URL was blocked by security policy', 'The result URL cloud parsing returned points at an internal or illegal address and was refused for safety. This is a security-policy outcome, not a network fault.', ''],
        mineru_zip_bomb: ['The cloud result archive is abnormal', 'The archive cloud parsing returned has an abnormal expansion ratio and was refused for safety.', ''],
        mineru_bad_zip: ['The cloud result is not a valid archive', 'What cloud parsing returned is not a readable archive. The same file uploaded again will not change the result.', ''],
        mineru_bad_json: ['The cloud result is not valid JSON', 'The index file cloud parsing returned could not be parsed as JSON.', ''],
        mineru_bad_json_shape: ['The cloud result has the wrong shape', 'The JSON cloud parsing returned does not match the agreed structure.', ''],
        mineru_bad_code: ['Cloud parsing returned an abnormal status', 'The status code cloud parsing returned is outside the agreed range.', ''],
        mineru_bad_content_list: ['The cloud result has no content list', 'What cloud parsing returned has no content list, so there is no usable parse result.', ''],
        mineru_empty_result: ['Cloud parsing produced no content', 'Cloud parsing completed but produced no content. This document has no result on this path.', ''],
        mineru_incomplete_batch: ['The cloud parsing batch is incomplete', 'Cloud parsing returned only part of the batch; the remaining pages were not parsed, so the overall result is unusable.', ''],
        mineru_local_read_error: ['Could not read the cloud result file', 'The result was downloaded but could not be read locally. Please check the server’s disk permissions.', ''],
        mineru_response_too_large: ['The cloud parsing response is over the limit', 'What the cloud parsing service returned exceeds the server’s size limit and was refused.', ''],
        mineru_missing_result_url: ['Cloud parsing gave no result URL', 'The cloud parsing service did not return the location of a result file, so no result was obtained.', ''],
        mineru_empty_response: ['Cloud parsing returned nothing', 'The cloud parsing service returned an empty response. You can retry.', ''],
        mineru_network_error: ['Cloud parsing network error', 'The network was down while calling the cloud parsing service. You can retry.', ''],
        mineru_http_error: ['The cloud parsing service returned an error', 'The cloud parsing service returned an error status. You can retry later.', ''],
        mineru_business_error: ['Cloud parsing business error', 'The cloud parsing service reported a business error (the file was not accepted, or the task parameters are wrong). You can try once more.', ''],
        mineru_task_failed: ['The cloud parsing task failed', 'The cloud parsing task failed to execute. You can retry.', ''],
        archive_bomb: ['The archive expands to too much', 'This archive’s expanded volume is abnormal and was refused for safety.', ''],
        archive_nested_bomb: ['The nested archive expands to too much', 'An archive inside this archive expands to an abnormal volume and was refused for safety.', ''],
        archive_nested_too_deep: ['The archive is nested too deeply', 'This archive’s nesting depth exceeds the server’s limit.', ''],
        archive_traversal: ['The archive contains out-of-bounds paths', 'A path inside this archive tries to write outside the extraction directory and was refused for safety.', ''],
        archive_symlink: ['The archive contains symbolic links', 'This archive contains a link pointing outside the extraction directory and was refused for safety.', ''],
        consent_not_granted: ['Cloud parsing is not authorised', 'The privacy policy allows cloud parsing, but your explicit consent has not been given. The document was not uploaded and parsing did not start.', 'Tick the cloud parsing consent option and try again.'],
        privacy_mode_local_only: ['Cloud parsing is blocked by the privacy policy', 'The current document privacy mode forbids any document leaving the machine. The document was not uploaded and parsing did not start.', 'This needs an administrator to set DOCUMENT_PRIVACY_MODE to allow_cloud.'],
        page_render_unavailable: ['This build cannot render page images', 'This deployment has no optional rendering dependency installed, so the original rendered image of this page cannot be produced. The evidence panel rebuilds the in-page layout from the parsed element coordinates instead, and marks clearly that it is not a rendered image.', 'Installing the optional rendering dependency makes this endpoint start returning page images.'],
        page_render_unsupported_format: ['Page rendering does not support this format', 'The renderer only handles document formats it knows, and this document is not one of them.', ''],
        page_render_failed: ['Page rendering failed', 'Rendering this page hit an error, so no image was produced. The source passage is still viewable.', ''],
        client_timeout: ['The request timed out', 'No reply arrived within the wait limit and the request was aborted. The server may still be working on it.', 'A long job does not stop because this timed out; reconnect to see how far it got.'],
        client_unreachable: ['Cannot reach the server', 'This address did not respond: the network may be down, the service may not be running, or the address may point somewhere else.', ''],
        client_offline: ['Currently offline', 'The browser reports that this device has no network connection. Queued operations go out once the connection is back.', ''],
        client_malformed_response: ['The response could not be parsed', 'What the server returned is not the format this endpoint specifies. The rest of the page is unaffected.', 'This usually means the server version and this page disagree.'],
        client_response_too_large: ['The response is over the size limit', 'What the server returned exceeds the read limit this page sets, and reading stopped partway.', ''],
        client_stream_interrupted: ['The connection dropped', 'The data stream broke partway through. The server may already have finished this work, so reconnecting confirms it rather than redoing it.', ''],
        client_cancelled: ['Cancelled', 'This operation was cancelled and produced no result.', ''],
        client_unknown_error: ['An unclassified error occurred', 'This operation failed, but the cause came neither from the server response nor from the network. Nothing half-finished was left on the page.', 'Please tell operations the steps you took at the time.']
    }
};

/** Every code this build can present, in both languages. Used by the tests. */
export const ERROR_CATALOGUE = ERROR_COPY;

/**
 * The `{title, message, hint}` a reader should see for a backend `code`.
 *
 * Deliberately *not* a parameter of the backend's own `error` sentence. The
 * lookup is by `code`; the sentence is returned separately as `serverDetail`
 * so the caller can decide whether showing it adds anything (it usually does
 * not, because it is in a language we do not know the reader reads).
 *
 * An unknown code falls back to {@link UNKNOWN_ERROR_PRESENTATION} rather than
 * to the Chinese server prose, because "a code this build does not know" is
 * itself a thing worth saying, and saying it in the reader's language is
 * better than passing off a sentence from an unknown language as if it were
 * the interface's.
 */
export function errorPresentation(code, serverDetail) {
    const key = String(code || '').trim();
    const entry = (ERROR_COPY[currentLocale] || {})[key];
    const zh = ERROR_COPY['zh-CN'][key];
    const resolved = entry
        || (zh ? zh : null)
        || null;

    if (resolved) {
        const [title, message, hint] = resolved;
        return {
            code: key || 'client_unknown_error',
            known: true,
            title,
            message,
            hint: hint || '',
            serverDetail: String(serverDetail || '').trim()
        };
    }

    const unknown = UNKNOWN_ERROR_PRESENTATION;
    return {
        code: key || 'client_unknown_error',
        known: false,
        title: t('error.unknown.title'),
        message: t('error.unknown.message'),
        hint: t('error.unknown.hint'),
        serverDetail: String(serverDetail || '').trim(),
        // The raw code is what an operator needs; the untranslated title is
        // kept as a peer so nothing is silently discarded.
        unknownTitle: unknown.title
    };
}

// ==========================================================================
//
// The interface language vs. the document's language
// ==========================================================================

/**
 * The language of *document content*, decided from the content itself.
 *
 * This is a separate axis from the interface language and must never be derived
 * from it. The corpus is Chinese and English papers, frequently mixed inside
 * one file; a reader who switches the interface to English is still reading a
 * Chinese paper, and labelling that paper's sections “Summary” while
 * translating its headings would relabel the document rather than the tool.
 *
 * `lang` on a content region is therefore derived from the text inside it.
 * A `lang` that disagrees with the words it wraps is worse than no `lang`: it
 * makes a screen reader pronounce Chinese with English phonetics.
 */
export function documentLocale(sample) {
    const text = String(sample || '');
    if (!text.trim()) return null;
    let cjk = 0;
    for (const character of text) {
        const code = character.codePointAt(0);
        if (
            (code >= 0x4e00 && code <= 0x9fff)   // CJK Unified Ideographs
            || (code >= 0x3040 && code <= 0x30ff) // kana
            || (code >= 0xac00 && code <= 0xd7af) // hangul
            || (code >= 0xf900 && code <= 0xfaff) // compatibility ideographs
        ) {
            cjk += 1;
        }
    }
    // A handful of ideographs in an English abstract is a citation, not a
    // document language; the ratio keeps one quoted name from flipping it.
    return (cjk / text.length) > 0.2 ? 'zh-CN' : 'en';
}

/**
 * Stamp a content region with the language of what it contains.
 *
 * Callers pass the *document's* text, never the interface locale. A region
 * whose language cannot be decided is left unlabelled rather than labelled
 * wrongly.
 */
export function markDocumentLanguage(element, sample) {
    if (!element || !element.setAttribute) return;
    const detected = documentLocale(sample);
    if (detected) {
        element.setAttribute('lang', detected);
    } else {
        element.removeAttribute('lang');
    }
}

// ==========================================================================
//
// Applying a locale to the document
// ==========================================================================

/**
 * Set `<html lang>` and the companion hooks, and translate the static markup.
 *
 * `lang` is the minimum; the rest matters because a page whose `lang` says
 * Chinese while the visible words are English misinforms a screen reader, a
 * browser's own translate prompt, and `:lang()` rules in the stylesheet. The
 * `dir` attribute is set from the locale rather than left to inherit, so
 * adding a right-to-left language later is a catalogue entry and not a
 * stylesheet change.
 */
export function applyLocale(locale) {
    const next = normalizeLocale(locale) || DEFAULT_LOCALE;
    currentLocale = next;

    if (typeof document !== 'undefined' && document.documentElement) {
        const root = document.documentElement;
        root.setAttribute('lang', next);
        root.setAttribute('dir', next === 'ar' || next === 'he' || next === 'fa' ? 'rtl' : 'ltr');
        // A hook the stylesheet can key on, distinct from `lang`: this one
        // means "the interface is in this language", which is the question
        // font stacks and letter-spacing actually need answered.
        root.setAttribute('data-ui-lang', next.split('-')[0]);
    }

    translateStaticMarkup();
    updateSwitcher();
    notifyLocaleChange(next);
}

/**
 * Translate every `data-i18n` in the document.
 *
 * Static markup only. Anything rendered by a module goes through `t()` at
 * render time and is re-rendered by its own module on
 * `paperwhisperer:localechange`; see `onLocaleChange`.
 *
 * `data-i18n-attr` carries the attribute-driven strings in one attribute:
 * `data-i18n-attr="aria-label:switcher.label;title:switcher.title"` — one
 * attribute instead of N, so an element never ends up with a label whose
 * half-translated attributes disagree with its half-translated text.
 */
export function translateStaticMarkup(root) {
    if (typeof document === 'undefined') return;
    const scope = root || document;

    scope.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        if (key) element.textContent = t(key);
    });

    scope.querySelectorAll('[data-i18n-attr]').forEach(element => {
        const pairs = element.getAttribute('data-i18n-attr') || '';
        pairs.split(';').forEach(pair => {
            const [attribute, key] = pair.split(':').map(part => part.trim());
            if (attribute && key) element.setAttribute(attribute, t(key));
        });
    });
}

function notifyLocaleChange(locale) {
    localeListeners.forEach(listener => {
        try {
            listener(locale);
        } catch (error) {
            // One bad subscriber must not leave the switcher half-applied.
            console.warn('Locale listener failed:', error);
        }
    });
    if (typeof document !== 'undefined' && typeof CustomEvent === 'function') {
        document.dispatchEvent(new CustomEvent('paperwhisperer:localechange', {
            detail: { locale }
        }));
    }
}

/**
 * Subscribe to locale changes. Returns an unsubscribe function.
 *
 * Modules that render their own text need this: their strings are produced at
 * render time, so a locale change has to be followed by a re-render, and the
 * only honest way for `i18n.js` to trigger that without importing every
 * feature module (and creating an import cycle through them) is an event.
 */
export function onLocaleChange(listener) {
    localeListeners.add(listener);
    return () => localeListeners.delete(listener);
}

// ==========================================================================
//
// The switcher
// ==========================================================================

/** Endonyms, so each option names its own language in that language. */
const LOCALE_NAMES = {
    'zh-CN': '中文',
    en: 'English'
};

function updateSwitcher() {
    if (typeof document === 'undefined') return;
    const group = document.getElementById('languageSwitcher');
    if (!group) return;

    group.querySelectorAll('[data-locale-option]').forEach(option => {
        const locale = normalizeLocale(option.getAttribute('data-locale-option'));
        const selected = locale === currentLocale;
        // `aria-checked` is what a screen reader announces for a radio; the
        // visible state is carried by `data-selected` as well, so the control
        // is never colour-only.
        option.setAttribute('aria-checked', String(selected));
        option.setAttribute('data-selected', String(selected));
        option.tabIndex = selected ? 0 : -1;
    });
}

function handleSwitcherKeydown(event) {
    const options = Array.from(event.currentTarget.querySelectorAll('[data-locale-option]'));
    const index = options.indexOf(document.activeElement);
    if (index < 0) return;

    // A radiogroup is one tab stop, so the arrow keys move within it. Without
    // this the control is reachable but the reader cannot tell it is a group,
    // and Left/Right scroll the page instead.
    let next = null;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
        next = options[(index + 1) % options.length];
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
        next = options[(index - 1 + options.length) % options.length];
    } else if (event.key === ' ' || event.key === 'Enter') {
        event.preventDefault();
        setLocale(options[index].getAttribute('data-locale-option'));
        return;
    }
    if (!next) return;
    event.preventDefault();
    next.focus();
    setLocale(next.getAttribute('data-locale-option'));
}

/**
 * Switch language and remember the choice.
 *
 * The choice is persisted *before* the interface is re-rendered, so a storage
 * failure costs the reader their preference for this page view and nothing
 * more — never a half-applied switch. `applyLocale` is what notifies, so a
 * failed write does not produce two events.
 */
export function setLocale(locale) {
    const next = normalizeLocale(locale);
    if (!next) return false;
    if (next === currentLocale && readStoredLocale() === next) return true;
    writeStoredLocale(next);
    applyLocale(next);
    return true;
}

/**
 * Wire the switcher and set the starting language. Idempotent.
 *
 * Called from `main.js` before any feature module renders, so the first paint
 * is already in the reader's language rather than flashing the default and
 * swapping.
 */
export function initializeI18n() {
    if (typeof document === 'undefined') return;

    const group = document.getElementById('languageSwitcher');
    if (group && !group.dataset.i18nBound) {
        group.dataset.i18nBound = 'true';
        group.addEventListener('keydown', handleSwitcherKeydown);
        group.querySelectorAll('[data-locale-option]').forEach(option => {
            option.addEventListener('click', () => {
                setLocale(option.getAttribute('data-locale-option'));
            });
        });
        group.setAttribute('data-locale-count', String(SUPPORTED_LOCALES.length));
    }

    applyLocale(resolveInitialLocale());
}

export { LOCALE_NAMES, STORAGE_KEY };
