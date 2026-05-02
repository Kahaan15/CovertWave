var API_BASE = 'http://localhost:8000/api';

// ─────────────────────────────────────────────────────────────────────────────
// Toast Notifications
// ─────────────────────────────────────────────────────────────────────────────

function showToast(message, isError) {
    var container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }
    var toast = document.createElement('div');
    toast.className = 'toast ' + (isError ? 'error' : 'success');
    var icon = document.createElement('span');
    icon.textContent = isError ? '!' : '\u2713';
    var text = document.createElement('span');
    text.textContent = message;
    toast.appendChild(icon);
    toast.appendChild(text);
    container.appendChild(toast);
    setTimeout(function() {
        toast.classList.add('fade-out');
        setTimeout(function() { toast.remove(); }, 300);
    }, 3000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Navigation
// ─────────────────────────────────────────────────────────────────────────────

document.querySelectorAll('.nav-links li').forEach(function(item) {
    item.addEventListener('click', function(e) {
        document.querySelectorAll('.nav-links li').forEach(function(nav) { nav.classList.remove('active'); });
        e.target.classList.add('active');
        var targetId = e.target.getAttribute('data-target');
        document.querySelectorAll('.view-section').forEach(function(sec) { sec.classList.add('hidden'); });
        document.getElementById(targetId).classList.remove('hidden');
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// Shared Helpers
// ─────────────────────────────────────────────────────────────────────────────

function toggleLoading(btn, isLoading) {
    var text = btn.querySelector('.btn-text');
    var loader = btn.querySelector('.loader');
    if (isLoading) {
        text.style.display = 'none';
        loader.classList.remove('hidden');
        btn.disabled = true;
    } else {
        text.style.display = 'block';
        loader.classList.add('hidden');
        btn.disabled = false;
    }
}

function updateFileLabel(input) {
    var label = input.nextElementSibling;
    if (!label) return;
    var nameSpan = label.querySelector('.file-name');
    if (!nameSpan) return;
    if (input.files && input.files.length > 0) {
        nameSpan.textContent = input.files[0].name;
        label.style.borderColor = 'var(--primary)';
    } else {
        nameSpan.textContent = 'Choose WAV File';
        label.style.borderColor = '';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Encode Tab
// ─────────────────────────────────────────────────────────────────────────────

var encAudioInput = document.getElementById('enc-audio');
var encLsbSelect = document.getElementById('enc-lsb');
var capacityDisplay = document.getElementById('capacity-display');
var encMessage = document.getElementById('enc-message');
var meterContainer = document.getElementById('capacity-meter-container');
var meterText = document.getElementById('capacity-meter-text');
var meterFill = document.getElementById('capacity-meter-fill');
var currentMaxCapacity = 0;

function updateCapacityMeter() {
    if (currentMaxCapacity <= 0) return;
    var currentLen = encMessage.value.length;
    var percentage = (currentLen / currentMaxCapacity) * 100;
    meterText.textContent = Math.round(percentage) + '% (' + currentLen + ' / ' + currentMaxCapacity + ' chars)';
    if (percentage > 100) percentage = 100;
    meterFill.style.width = percentage + '%';
    meterFill.classList.remove('warning', 'danger');
    if (percentage > 65) {
        meterFill.classList.add('danger');
    } else if (percentage >= 25) {
        meterFill.classList.add('warning');
    }
}

encMessage.addEventListener('input', updateCapacityMeter);

function resetEncodeForm() {
    encMessage.value = '';
    document.getElementById('enc-password').value = '';
    document.getElementById('enc-algo').selectedIndex = 0;
    encLsbSelect.selectedIndex = 0;
    currentMaxCapacity = 0;
    capacityDisplay.innerHTML = 'Select an audio file to view capacity.';
    meterContainer.classList.add('hidden');
    meterFill.style.width = '0%';
    meterFill.classList.remove('warning', 'danger');
}

function checkCapacity() {
    if (!encAudioInput.files.length) return;
    var file = encAudioInput.files[0];
    var formData = new FormData();
    formData.append('file', file);
    formData.append('lsb_bits', encLsbSelect.value);
    capacityDisplay.innerHTML = 'Calculating capacity...';
    fetch(API_BASE + '/capacity', { method: 'POST', body: formData })
        .then(function(response) {
            if (!response.ok) throw new Error('Failed to calculate capacity');
            return response.json();
        })
        .then(function(data) {
            capacityDisplay.innerHTML =
                '<div>Total Samples: <span>' + data.total_samples + '</span></div>' +
                '<div>Max Hidden Message: <span>~' + data.message_capacity + ' characters</span></div>';
            currentMaxCapacity = data.message_capacity;
            meterContainer.classList.remove('hidden');
            updateCapacityMeter();
        })
        .catch(function(error) {
            capacityDisplay.innerHTML = '<span style="color: #ef4444">Error: ' + error.message + '</span>';
        });
}

// Single handler: reset first, then check capacity
encAudioInput.addEventListener('change', function() {
    updateFileLabel(this);
    resetEncodeForm();
    checkCapacity();
});

encLsbSelect.addEventListener('change', checkCapacity);

// Encode submit
document.getElementById('encode-form').addEventListener('submit', function(e) {
    e.preventDefault();
    var btn = e.target.querySelector('button');
    toggleLoading(btn, true);

    var formData = new FormData();
    formData.append('file', document.getElementById('enc-audio').files[0]);
    formData.append('message', document.getElementById('enc-message').value);
    formData.append('password', document.getElementById('enc-password').value);
    formData.append('algorithm', document.getElementById('enc-algo').value);
    formData.append('lsb_bits', document.getElementById('enc-lsb').value);

    fetch(API_BASE + '/encode', { method: 'POST', body: formData })
        .then(function(response) {
            if (!response.ok) return response.json().then(function(err) { throw new Error(err.detail || 'Encoding failed'); });
            return response.blob();
        })
        .then(function(blob) {
            var url = window.URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'stego_' + document.getElementById('enc-audio').files[0].name;
            document.body.appendChild(a);
            a.click();
            a.remove();
            showToast('Encoding successful! File downloaded.', false);
        })
        .catch(function(error) {
            showToast(error.message, true);
        })
        .finally(function() {
            toggleLoading(btn, false);
        });
});

// ─────────────────────────────────────────────────────────────────────────────
// Decode Tab
// ─────────────────────────────────────────────────────────────────────────────

var payloadChartInstance = null;

function resetDecodeForm() {
    document.getElementById('dec-password').value = '';
    document.getElementById('dec-algo').selectedIndex = 0;
    document.getElementById('dec-lsb').selectedIndex = 0;
    document.getElementById('decode-result').classList.add('hidden');
    document.getElementById('decrypted-text').textContent = '';
    document.getElementById('extraction-summary').classList.add('hidden');
    document.getElementById('extraction-summary').innerHTML = '';
    if (payloadChartInstance) {
        payloadChartInstance.destroy();
        payloadChartInstance = null;
    }
}

document.getElementById('dec-audio').addEventListener('change', function() {
    updateFileLabel(this);
    resetDecodeForm();
});

document.getElementById('decode-form').addEventListener('submit', function(e) {
    e.preventDefault();
    var btn = e.target.querySelector('button');
    var resultBox = document.getElementById('decode-result');
    var resultText = document.getElementById('decrypted-text');

    toggleLoading(btn, true);
    resultBox.classList.add('hidden');
    resultText.textContent = '';

    var formData = new FormData();
    formData.append('file', document.getElementById('dec-audio').files[0]);
    formData.append('password', document.getElementById('dec-password').value);
    formData.append('algorithm', document.getElementById('dec-algo').value);
    formData.append('lsb_bits', document.getElementById('dec-lsb').value);

    fetch(API_BASE + '/decode', { method: 'POST', body: formData })
        .then(function(response) {
            if (!response.ok) return response.json().then(function(err) { throw new Error(err.detail || 'Decoding failed. Incorrect password or file.'); });
            return response.json();
        })
        .then(function(data) {
            resultText.textContent = data.message;
            resultBox.classList.remove('hidden');
            if (data.energy_profile && data.payload_density) {
                renderPayloadChart(data.energy_profile, data.payload_density, data.message);
            }
        })
        .catch(function(error) {
            showToast(error.message, true);
        })
        .finally(function() {
            toggleLoading(btn, false);
        });
});

// ─────────────────────────────────────────────────────────────────────────────
// Analyze Tab
// ─────────────────────────────────────────────────────────────────────────────

function resetAnalyzeResults() {
    var dashboard = document.getElementById('analyze-results');
    dashboard.classList.add('hidden');
    dashboard.innerHTML = '';
}

document.getElementById('ana-audio').addEventListener('change', function() {
    updateFileLabel(this);
    resetAnalyzeResults();
});

document.getElementById('analyze-form').addEventListener('submit', function(e) {
    e.preventDefault();
    var btn = e.target.querySelector('button');
    var dashboard = document.getElementById('analyze-results');

    toggleLoading(btn, true);
    dashboard.classList.add('hidden');
    dashboard.innerHTML = '';

    var formData = new FormData();
    formData.append('file', document.getElementById('ana-audio').files[0]);

    fetch(API_BASE + '/analyze', { method: 'POST', body: formData })
        .then(function(response) {
            if (!response.ok) return response.json().then(function(err) { throw new Error(err.detail || 'Analysis failed'); });
            return response.json();
        })
        .then(function(data) {
            renderAnalysisDashboard(data);
            dashboard.classList.remove('hidden');
        })
        .catch(function(error) {
            showToast(error.message, true);
        })
        .finally(function() {
            toggleLoading(btn, false);
        });
});

// ─────────────────────────────────────────────────────────────────────────────
// Renderers
// ─────────────────────────────────────────────────────────────────────────────

function renderAnalysisDashboard(data) {
    var dashboard = document.getElementById('analyze-results');
    var summary = data.detection_summary;
    var isDetected = summary.methods_triggered > 1;
    var mainVerdictClass = isDetected ? 'detected' : 'clean';
    var mainVerdictText = isDetected ? 'STEGO DETECTED' : (summary.methods_triggered === 1 ? 'SUSPICIOUS' : 'CLEAN');

    var html = '<div class="metric-card ' + mainVerdictClass + '">' +
        '<div class="metric-label">Overall Verdict</div>' +
        '<div class="metric-value">' + mainVerdictText + '</div>' +
        '<div class="metric-label">' + summary.methods_triggered + ' / 4 Methods Triggered</div>' +
        '</div>' +
        '<div class="glass-panel" style="margin-top: 1rem">' +
        '<h3>Detailed Battery Results</h3>' +
        '<table class="matrix-table"><thead><tr><th>Method</th><th>Verdict</th><th>Status</th></tr></thead><tbody>';

    var methods = [
        { name: 'Chi-Square Test', data: data.chi_square },
        { name: 'RS Analysis', data: data.rs_analysis },
        { name: 'Sample Pair Analysis', data: data.spa },
        { name: 'Histogram Analysis', data: data.histogram }
    ];

    methods.forEach(function(m) {
        var badgeClass = m.data.detected ? 'danger' : 'success';
        var badgeText = m.data.detected ? 'DETECTED' : 'CLEAN';
        html += '<tr><td>' + m.name + '</td>' +
            '<td style="color: var(--text-muted); font-size: 0.9rem">' + m.data.verdict + '</td>' +
            '<td><span class="badge ' + badgeClass + '">' + badgeText + '</span></td></tr>';
    });

    html += '</tbody></table></div>';
    dashboard.innerHTML = html;
}

function renderPayloadChart(energyData, densityData, message) {
    var ctx = document.getElementById('payloadChart').getContext('2d');

    if (payloadChartInstance) {
        payloadChartInstance.destroy();
    }

    var totalBits = densityData.reduce(function(a, b) { return a + b; }, 0);
    var summaryBox = document.getElementById('extraction-summary');
    summaryBox.innerHTML = '<h4>Secure Extraction Summary</h4>' +
        'Your decrypted message is <strong>' + message.length + ' characters</strong> long. ' +
        'However, to guarantee strong security, CovertWave encrypted your message into a secure ' +
        '<strong>' + totalBits + '-bit payload</strong> using AES-256. This payload was shattered into ' +
        'microscopic fragments and seamlessly embedded across the audio file. The neon green spikes on ' +
        'the waveform above reveal the exact physical locations of these hidden fragments.';
    summaryBox.classList.remove('hidden');

    var labels = energyData.map(function(_, i) { return i; });

    payloadChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Encrypted Footprint',
                    data: densityData,
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.4)',
                    borderWidth: 2,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.2,
                    yAxisID: 'y1'
                },
                {
                    label: 'Audio Energy',
                    data: energyData,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.15)',
                    borderWidth: 1,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: { display: false },
                y: { type: 'linear', display: false, position: 'left' },
                y1: { type: 'linear', display: false, position: 'right', grid: { drawOnChartArea: false } }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.datasetIndex === 0 ? 'Encrypted Data Fragment' : 'Clean Audio Wave';
                        },
                        title: function() { return null; }
                    }
                },
                legend: { labels: { color: '#cbd5e1' } }
            }
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Password Toggle (hold to reveal)
// ─────────────────────────────────────────────────────────────────────────────

document.querySelectorAll('.password-toggle').forEach(function(toggle) {
    var input = toggle.previousElementSibling;
    toggle.addEventListener('mousedown', function() { input.type = 'text'; });
    toggle.addEventListener('mouseup', function() { input.type = 'password'; });
    toggle.addEventListener('mouseleave', function() { input.type = 'password'; });
});
