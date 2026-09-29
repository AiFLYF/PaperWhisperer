import { MAX_UPLOAD_BYTES, SUPPORTED_UPLOAD_EXTENSIONS } from './config.js';
import { hideError, showError } from './dom.js';

/** File picking, drag-and-drop, upload validation and the file info bar. */

function initializeUploadDropZone() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('file');
    if (!dropZone || !fileInput) return;

    dropZone.addEventListener('click', event => {
        if (event.target !== fileInput) fileInput.click();
    });
    dropZone.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            fileInput.click();
        }
    });
    ['dragenter', 'dragover'].forEach(type => {
        dropZone.addEventListener(type, event => {
            event.preventDefault();
            dropZone.classList.add('drag-over');
        });
    });
    ['dragleave', 'drop'].forEach(type => {
        dropZone.addEventListener(type, event => {
            event.preventDefault();
            dropZone.classList.remove('drag-over');
        });
    });
    dropZone.addEventListener('drop', event => {
        const file = event.dataTransfer?.files?.[0];
        if (!file) return;
        const transfer = new DataTransfer();
        transfer.items.add(file);
        fileInput.files = transfer.files;
        handleFileSelection();
    });
}

function getUploadValidationError(file) {
    if (!file) return 'Please select a document first.';
    const lowerName = file.name.toLowerCase();
    const isSupported = SUPPORTED_UPLOAD_EXTENSIONS.some(extension => lowerName.endsWith(extension));
    if (!isSupported) {
        return `Unsupported file type. Please upload ${SUPPORTED_UPLOAD_EXTENSIONS.join(', ')}.`;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
        return `File is too large (${formatFileSize(file.size)}). Maximum supported size is ${formatFileSize(MAX_UPLOAD_BYTES)}.`;
    }
    return '';
}

function handleFileSelection() {
    const fileInput = document.getElementById('file');
    const file = fileInput?.files?.[0];
    const error = getUploadValidationError(file);
    updateFileMeta(error);
    if (error && file) showError(error);
    if (!error) hideError();
}

function formatFileSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let index = 0;
    while (size >= 1024 && index < units.length - 1) {
        size /= 1024;
        index += 1;
    }
    return `${size.toFixed(size >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function getUploadFileExtension(fileName) {
    const dotIndex = fileName.lastIndexOf('.');
    return dotIndex >= 0 ? fileName.slice(dotIndex).toUpperCase() : 'Unknown';
}

function appendFileMetaItem(container, label, value, extraClass = '') {
    const item = document.createElement('span');
    item.className = `file-meta-item${extraClass ? ` ${extraClass}` : ''}`;

    const itemLabel = document.createElement('span');
    itemLabel.className = 'file-meta-label';
    itemLabel.textContent = label;

    const itemValue = document.createElement('strong');
    itemValue.textContent = value;

    item.append(itemLabel, itemValue);
    container.appendChild(item);
}

function updateFileMeta(validationError = '') {
    const fileInput = document.getElementById('file');
    const file = fileInput?.files?.[0];
    const fileMeta = document.getElementById('fileMeta');
    const dropZone = document.getElementById('dropZone');
    if (!fileMeta) return;

    fileMeta.classList.remove('file-meta-ready', 'file-meta-error');
    fileMeta.replaceChildren();

    if (!file) {
        fileMeta.textContent = 'No file selected. Recommended: clean PDF, TXT, DOCX, or PPTX for better structure extraction.';
        dropZone?.classList.remove('has-file', 'has-error');
        dropZone?.setAttribute('aria-invalid', 'false');
        return;
    }

    const hasError = Boolean(validationError);
    dropZone?.classList.toggle('has-file', !hasError);
    dropZone?.classList.toggle('has-error', hasError);
    dropZone?.setAttribute('aria-invalid', String(hasError));
    fileMeta.classList.add(hasError ? 'file-meta-error' : 'file-meta-ready');

    const summary = document.createElement('div');
    summary.className = 'file-meta-summary';

    const status = document.createElement('span');
    status.className = 'file-meta-status';
    status.textContent = hasError ? 'Review needed' : 'Ready to analyze';

    const guidance = document.createElement('span');
    guidance.textContent = hasError ? validationError : 'Supported document type and size.';

    summary.append(status, guidance);

    const details = document.createElement('div');
    details.className = 'file-meta-details';
    appendFileMetaItem(details, 'Name', file.name, 'file-meta-name');
    appendFileMetaItem(details, 'Size', formatFileSize(file.size));
    appendFileMetaItem(details, 'Type', getUploadFileExtension(file.name));
    appendFileMetaItem(details, 'Limit', formatFileSize(MAX_UPLOAD_BYTES));

    fileMeta.append(summary, details);
}

function setFileInfo(fileName, charCount) {
    const fileInfo = document.getElementById('fileInfo');
    fileInfo.replaceChildren();

    const left = document.createElement('span');
    left.textContent = 'Document: ';
    const strong = document.createElement('strong');
    strong.textContent = fileName;
    left.appendChild(strong);

    const actions = document.createElement('div');
    actions.className = 'file-info-actions';

    const stats = document.createElement('span');
    stats.textContent = `Tokens/Chars: ${charCount || 'N/A'}`;

    actions.appendChild(stats);
    fileInfo.appendChild(left);
    fileInfo.appendChild(actions);
}


export { formatFileSize };
export { getUploadFileExtension };
export { getUploadValidationError };
export { handleFileSelection };
export { initializeUploadDropZone };
export { setFileInfo };
export { updateFileMeta };
