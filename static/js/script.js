document.addEventListener('DOMContentLoaded', () => {

    /* --- Scroll Reveal Animations --- */
    const revealElements = document.querySelectorAll('[data-reveal]');
    
    const revealOptions = {
        threshold: 0.15,
        rootMargin: "0px 0px -50px 0px"
    };

    const revealOnScroll = new IntersectionObserver(function(entries, observer) {
        entries.forEach(entry => {
            if (!entry.isIntersecting) {
                return;
            } else {
                entry.target.classList.add('revealed');
                
                // If it has a counter, trigger it when revealed
                const counter = entry.target.querySelector('[data-count]');
                if (counter && !counter.classList.contains('counted')) {
                    animateCounter(counter);
                }
                
                // Also check if the element itself is a counter
                if (entry.target.hasAttribute('data-count') && !entry.target.classList.contains('counted')) {
                    animateCounter(entry.target);
                }

                observer.unobserve(entry.target);
            }
        });
    }, revealOptions);

    revealElements.forEach(el => {
        revealOnScroll.observe(el);
    });

    /* --- Animated Counters --- */
    function animateCounter(el) {
        el.classList.add('counted');
        const target = +el.getAttribute('data-count');
        const duration = 2000; // 2 seconds
        const stepTime = Math.abs(Math.floor(duration / target));
        let current = 0;
        
        const timer = setInterval(() => {
            current += 1;
            el.innerText = current;
            if (current >= target) {
                el.innerText = target;
                clearInterval(timer);
            }
        }, stepTime === 0 ? 10 : stepTime);
    }

    // Trigger counters that might be above the fold or not marked with data-reveal
    document.querySelectorAll('[data-count]:not([data-reveal])').forEach(counter => {
        animateCounter(counter);
    });


    /* --- File Upload Interactions --- */
    const dropzone = document.querySelector('.upload-dropzone');
    const fileInput = document.querySelector('[data-file-input]');
    const fileNameDisplay = document.querySelector('#selectedFileName');
    const imagePreview = document.querySelector('#imagePreview');
    const emptyPreview = document.querySelector('#previewEmpty');

    if (dropzone && fileInput) {
        // Drag over effects
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
            dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
            dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length) {
                fileInput.files = files;
                updateFileDisplay(files[0]);
            }
        });

        fileInput.addEventListener('change', function() {
            if (this.files && this.files.length > 0) {
                updateFileDisplay(this.files[0]);
            }
        });

        function updateFileDisplay(file) {
            if (fileNameDisplay) {
                fileNameDisplay.textContent = file.name;
                fileNameDisplay.style.color = "var(--success)";
                fileNameDisplay.style.borderColor = "var(--success)";
            }

            if (file && file.type.startsWith('image/') && imagePreview) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.classList.remove('is-hidden');
                    if (emptyPreview) emptyPreview.classList.add('is-hidden');
                }
                reader.readAsDataURL(file);
            }
        }
    }

    /* --- Live Camera Integration Details --- */
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas ? canvas.getContext('2d') : null;
    const overlay = document.getElementById('overlay');
    const octx = overlay ? overlay.getContext('2d') : null;
    const btnStart = document.getElementById('btnStart');
    const btnStop = document.getElementById('btnStop');
    const chkSmooth = document.getElementById('chkSmooth');
    const camStatus = document.getElementById('camStatus');
    const streamState = document.getElementById('streamState');
    const outLabel = document.getElementById('outLabel');
    const outConf = document.getElementById('outConf');
    const confFill = document.getElementById('confFill');
    const outTop3 = document.getElementById('outTop3');
    const outErr = document.getElementById('outErr');

    if (btnStart && btnStop && video && canvas && overlay) {
        let stream = null;
        let liveActive = false;
        let loopPromise = null;
        const history = [];
        const HISTORY_MAX = 5;
        const FRAME_GAP_MS = 35;

        let lastCapW = 0;
        let lastCapH = 0;
        let overlayState = null;

        function setError(msg) {
            if (!outErr) return;
            if (!msg) {
                outErr.hidden = true;
                outErr.textContent = '';
                return;
            }
            outErr.hidden = false;
            outErr.textContent = msg;
        }

        function majorityLabel(labels) {
            const counts = {};
            for (const L of labels) { counts[L] = (counts[L] || 0) + 1; }
            let best = labels[labels.length - 1];
            let n = 0;
            for (const k in counts) {
                if (counts[k] > n) { n = counts[k]; best = k; }
            }
            return best;
        }

        function renderTop3(items) {
            if (!outTop3) return;
            outTop3.innerHTML = '';
            if (!items || !items.length) {
                outTop3.innerHTML = '<div style="text-align: center; color: rgba(255,255,255,0.2); padding: 1rem 0;">No active candidates</div>';
                return;
            }
            for (const it of items) {
                const li = document.createElement('li');
                li.className = 'rank-item';
                const confPercent = (it.confidence * 100).toFixed(1);
                li.innerHTML = `
                    <div class="rank-head">
                        <span>${it.label}</span>
                        <span style="font-variant-numeric: tabular-nums;">${confPercent}%</span>
                    </div>
                    <span class="rank-bar"><span style="width: ${confPercent}%;"></span></span>
                `;
                outTop3.appendChild(li);
            }
        }

        function syncOverlaySize() {
            const w = Math.max(1, Math.floor(video.clientWidth));
            const h = Math.max(1, Math.floor(video.clientHeight));
            overlay.width = w;
            overlay.height = h;
            drawOverlay();
        }

        function drawOverlay() {
            const ow = overlay.width;
            const oh = overlay.height;
            octx.clearRect(0, 0, ow, oh);
            if (!overlayState || !overlayState.bbox || !overlayState.iw || !overlayState.ih) return;
            const b = overlayState.bbox;
            const sx = ow / overlayState.iw;
            const sy = oh / overlayState.ih;
            const x = b.x * sx;
            const y = b.y * sy;
            const bw = b.width * sx;
            const bh = b.height * sy;
            
            octx.strokeStyle = '#10b981'; // success var
            octx.lineWidth = Math.max(2, Math.round(ow / 320));
            octx.strokeRect(x, y, bw, bh);
            
            const label = overlayState.displayLabel || overlayState.label || '';
            if (!label) return;
            const conf = overlayState.confidence;
            const sub = conf != null ? ' ' + (conf * 100).toFixed(0) + '%' : '';
            const text = (label + sub).trim();
            octx.font = '600 16px "Inter", system-ui, sans-serif';
            const pad = 8;
            const tw = Math.min(ow - x - pad, octx.measureText(text).width + pad * 2);
            const th = 28;
            let ty = y - th - 4;
            if (ty < 4) ty = y + bh + 4;
            octx.fillStyle = 'rgba(10, 15, 25, 0.9)';
            octx.fillRect(x, ty, tw, th);
            octx.fillStyle = '#f8fafc';
            octx.fillText(text, x + pad, ty + 20);
        }

        async function captureAndPredict() {
            if (!stream || video.readyState < 2) return;
            const w = video.videoWidth;
            const h = video.videoHeight;
            if (!w || !h) return;
            const capW = Math.min(w, 480);
            canvas.width = capW;
            canvas.height = Math.round(h * (capW / w));
            lastCapW = canvas.width;
            lastCapH = canvas.height;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.62);
            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl })
                });
                const data = await res.json();
                if (!data.ok) {
                    setError(data.error || 'Request failed');
                    overlayState = null;
                    drawOverlay();
                    return;
                }
                setError('');
                if (!data.hand_detected) {
                    if(outLabel) outLabel.textContent = 'Awaiting Subject';
                    if(outConf) outConf.textContent = data.message || 'No hand detected';
                    if(confFill) confFill.style.width = "0%";
                    renderTop3([]);
                    overlayState = null;
                    drawOverlay();
                    return;
                }
                let label = data.label;
                if (chkSmooth && chkSmooth.checked && label) {
                    history.push(label);
                    if (history.length > HISTORY_MAX) history.shift();
                    label = majorityLabel(history);
                } else {
                    history.length = 0;
                }
                
                if(outLabel) {
                    outLabel.textContent = label || '—';
                    outLabel.style.color = "var(--success)";
                }
                if(data.confidence != null) {
                    const confPercent = (data.confidence * 100).toFixed(1);
                    if(outConf) outConf.textContent = confPercent + '%';
                    if(confFill) confFill.style.width = confPercent + '%';
                }
                
                renderTop3(data.top3);
                overlayState = {
                    bbox: data.bbox,
                    iw: data.image_width || lastCapW,
                    ih: data.image_height || lastCapH,
                    label: data.label,
                    displayLabel: label,
                    confidence: data.confidence
                };
                drawOverlay();
            } catch (e) {
                setError(String(e));
                overlayState = null;
                drawOverlay();
            }
        }

        async function predictionLoop() {
            while (liveActive && stream) {
                await captureAndPredict();
                if (!liveActive) break;
                await new Promise((r) => setTimeout(r, FRAME_GAP_MS));
            }
        }

        btnStart.addEventListener('click', async () => {
            setError('');
            if (camStatus) camStatus.textContent = 'Starting camera…';
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: false
                });
                video.srcObject = stream;
                btnStart.disabled = true;
                btnStop.disabled = false;
                
                if (camStatus) camStatus.textContent = 'Active & Predicting';
                if (streamState) streamState.textContent = 'Streaming SECURE';
                video.style.filter = "grayscale(0%) contrast(100%)";
                video.style.opacity = "1";
                if (overlay) {
                    overlay.style.boxShadow = "inset 0 0 50px rgba(16, 185, 129, 0.2)";
                }
                
                const syncOnce = () => { requestAnimationFrame(syncOverlaySize); };
                video.addEventListener('loadedmetadata', syncOnce, { once: true });
                video.addEventListener('playing', syncOnce, { once: true });
                
                liveActive = true;
                loopPromise = predictionLoop();
            } catch (e) {
                if (camStatus) camStatus.textContent = 'Camera blocked or unavailable';
                setError(String(e));
            }
        });

        btnStop.addEventListener('click', () => {
            liveActive = false;
            loopPromise = null;
            history.length = 0;
            overlayState = null;
            if(octx) octx.clearRect(0, 0, overlay.width, overlay.height);
            if (stream) {
                stream.getTracks().forEach(t => t.stop());
                stream = null;
            }
            video.srcObject = null;
            btnStart.disabled = false;
            btnStop.disabled = true;
            
            if (camStatus) camStatus.textContent = 'Idle';
            if (streamState) streamState.textContent = 'Idle';
            video.style.filter = "grayscale(20%) contrast(120%)";
            video.style.opacity = "0.6";
            if (overlay) overlay.style.boxShadow = "none";
            
            if (outLabel) {
                outLabel.textContent = 'Awaiting Stream';
                outLabel.style.color = "var(--text-muted)";
            }
            if (outConf) outConf.textContent = '--%';
            if (confFill) confFill.style.width = "0%";
            renderTop3([]);
            setError('');
        });

        window.addEventListener('resize', () => {
            if (stream) syncOverlaySize();
        });
    }

});
