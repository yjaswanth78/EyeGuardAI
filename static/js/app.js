document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const btnCamera = document.getElementById('btn-camera');
    const fileUpload = document.getElementById('file-upload');
    const cameraView = document.getElementById('camera-view');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const btnCapture = document.getElementById('btn-capture');
    const btnCancelCamera = document.getElementById('btn-cancel-camera');
    
    const inputSection = document.getElementById('input-section');
    const resultSection = document.getElementById('result-section');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    const resultCard = document.getElementById('result-card');
    const resultTitle = document.getElementById('result-title');
    const resultConfidence = document.getElementById('result-confidence');
    const resultAdvice = document.getElementById('result-advice');
    const resultDisclaimer = document.getElementById('result-disclaimer');
    const resultIcon = document.getElementById('result-icon');
    
    const agentLogsList = document.getElementById('agent-logs-list');
    
    const btnReset = document.getElementById('btn-reset');
    const errorAlert = document.getElementById('error-alert');
    const errorText = document.getElementById('error-text');

    const historyList = document.getElementById('history-list');
    const btnClearHistory = document.getElementById('btn-clear-history');

    let stream = null;

    // --- Steps UI ---
    function setStep(stepNum) {
        document.querySelectorAll('.step').forEach((el, index) => {
            if (index + 1 === stepNum) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        });
    }

    // --- History Management ---
    let history = JSON.parse(localStorage.getItem('eyeguard_history')) || [];

    function saveToHistory(result) {
        history.unshift({
            date: new Date().toLocaleString(),
            category: result.category,
            title: result.title,
            confidence: result.confidence
        });
        if (history.length > 5) history.pop();
        localStorage.setItem('eyeguard_history', JSON.stringify(history));
        renderHistory();
    }

    function renderHistory() {
        if (history.length === 0) {
            historyList.innerHTML = '<p class="privacy-notice">No past scans available.</p>';
            btnClearHistory.classList.add('hidden');
            return;
        }
        btnClearHistory.classList.remove('hidden');
        historyList.innerHTML = history.map(item => `
            <div class="history-item">
                <div>
                    <strong>${item.title}</strong>
                    <div style="font-size: 0.8rem; color: #6b7280;">${item.date}</div>
                </div>
                <div style="font-weight: 600; color: ${item.category === 'NORMAL' ? 'var(--success)' : (item.category === 'RETAKE' || item.category === 'NOT_EYE' ? 'var(--warning)' : 'var(--danger)')}">
                    ${item.confidence}%
                </div>
            </div>
        `).join('');
    }

    btnClearHistory.addEventListener('click', () => {
        history = [];
        localStorage.removeItem('eyeguard_history');
        renderHistory();
    });

    renderHistory();

    // --- Helpers ---
    function showError(message) {
        errorText.textContent = message;
        errorAlert.classList.remove('hidden');
        setTimeout(() => errorAlert.classList.add('hidden'), 5000);
    }

    function resetUI() {
        inputSection.classList.remove('hidden');
        resultSection.classList.add('hidden');
        loadingSpinner.classList.add('hidden');
        fileUpload.value = '';
        setStep(1);
    }

    // --- Sample Images Logic ---
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const imgSrc = e.target.getAttribute('data-img');
            setStep(2);
            inputSection.classList.add('hidden');
            loadingSpinner.classList.remove('hidden');
            document.getElementById('loading-text').textContent = 'Agent Swarm Analyzing Sample...';
            
            try {
                const response = await fetch(imgSrc);
                const blob = await response.blob();
                const file = new File([blob], 'sample.jpg', { type: 'image/jpeg' });
                processImageFile(file);
            } catch (err) {
                console.error(err);
                loadingSpinner.classList.add('hidden');
                inputSection.classList.remove('hidden');
                showError("Failed to load sample image.");
                setStep(1);
            }
        });
    });


    // --- Camera Flow (Supports WebRTC + Mobile Native Camera Fallback) ---
    const cameraCaptureInput = document.getElementById('camera-capture-input');
    if (cameraCaptureInput) {
        cameraCaptureInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                processImageFile(e.target.files[0]);
            }
        });
    }

    btnCamera.addEventListener('click', async () => {
        // Modern mobile browsers block getUserMedia over plain HTTP on LAN IPs (http://192.168.x.x).
        // If not in a secure context (HTTPS/localhost), directly trigger native mobile camera!
        if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            if (cameraCaptureInput) {
                cameraCaptureInput.click();
                return;
            }
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            video.srcObject = stream;
            btnCamera.classList.add('hidden');
            document.querySelector('.upload-wrapper').classList.add('hidden');
            document.querySelector('.sample-images-section').classList.add('hidden');
            cameraView.classList.remove('hidden');
        } catch (err) {
            console.warn("getUserMedia unavailable, switching to native mobile camera:", err);
            if (cameraCaptureInput) {
                cameraCaptureInput.click();
            } else {
                showError("Camera access denied or unavailable.");
            }
        }
    });

    btnCancelCamera.addEventListener('click', () => {
        if (stream) stream.getTracks().forEach(track => track.stop());
        cameraView.classList.add('hidden');
        btnCamera.classList.remove('hidden');
        document.querySelector('.upload-wrapper').classList.remove('hidden');
        document.querySelector('.sample-images-section').classList.remove('hidden');
    });

    btnCapture.addEventListener('click', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        if (stream) stream.getTracks().forEach(track => track.stop());
        
        canvas.toBlob((blob) => {
            const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
            processImageFile(file);
        }, 'image/jpeg');
    });

    // --- File Upload Flow ---
    fileUpload.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            processImageFile(e.target.files[0]);
        }
    });

    // --- API Request ---
    async function processImageFile(file) {
        cameraView.classList.add('hidden');
        btnCamera.classList.remove('hidden');
        document.querySelector('.upload-wrapper').classList.remove('hidden');
        document.querySelector('.sample-images-section').classList.remove('hidden');

        inputSection.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
        
        let simInterval = simulateAgentThinking();
        
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            clearInterval(simInterval);

            if (!response.ok) throw new Error(data.error || "Analysis failed");

            displayResults(data);
        } catch (err) {
            clearInterval(simInterval);
            showError(err.message);
            resetUI();
        }
    }
    
    function simulateAgentThinking() {
        setStep(2);
        const states = [
            "Agent_QualityControl: Inspecting image parameters...",
            "Agent_QualityControl: Image structure valid. Dispatching...",
            "🔬 Custom_DualAttnNet: Running offline TFLite inference...",
            "🟡 Agent_LiverSpecialist: Measuring scleral yellow chromaticity (Icterus)...",
            "🩸 Agent_BloodSpecialist: Extracting periorbital tissue erythema & pallor...",
            "🫀 Agent_CardioSpecialist: Analyzing scleral micro-vessel caliber & density...",
            "Agent_ChiefMedicalOfficer: Synthesizing swarm reports & clinical consensus..."
        ];
        let i = 0;
        const textEl = document.getElementById('loading-text');
        return setInterval(() => {
            textEl.textContent = states[i];
            i = (i + 1) % states.length;
        }, 1200);
    }

    function displayResults(data) {
        loadingSpinner.classList.add('hidden');
        resultSection.classList.remove('hidden');
        setStep(3);

        resultTitle.textContent = data.title;
        resultConfidence.textContent = `${data.confidence}%`;
        resultAdvice.textContent = data.advice;
        resultDisclaimer.textContent = data.disclaimer || "";
        
        // Render Agent Logs
        agentLogsList.innerHTML = "";
        if (data.agent_logs && data.agent_logs.length > 0) {
            data.agent_logs.forEach(log => {
                const li = document.createElement('li');
                li.textContent = `[System] ${log}`;
                agentLogsList.appendChild(li);
            });
        } else {
            agentLogsList.innerHTML = "<li>[System] No anomalous logs generated. Swarm consensus nominal.</li>";
        }
        
        // Render Custom User Model
        if (data.user_model) {
            const predEl = document.getElementById('user-model-prediction');
            const probEl = document.getElementById('user-model-prob');
            const confEl = document.getElementById('user-model-conf');
            const badgeEl = document.getElementById('user-model-badge');
            
            if (predEl) {
                if (data.user_model.is_overruled) {
                    predEl.innerHTML = `${data.user_model.prediction} <span style="font-size:0.75rem; color:#d97706; font-weight:normal;">(Overruled by AI Swarm)</span>`;
                } else {
                    predEl.textContent = data.user_model.prediction || "Normal Eye";
                }
            }
            if (probEl) probEl.textContent = data.user_model.probability || "0.0%";
            if (confEl) confEl.textContent = `${data.user_model.confidence || 95}%`;
            
            if (badgeEl) {
                if (data.user_model.prediction === "Normal Eye") {
                    badgeEl.style.background = "var(--success)";
                    badgeEl.textContent = "Normal";
                } else if (data.user_model.is_overruled) {
                    badgeEl.style.background = "#d97706";
                    badgeEl.textContent = "Overruled";
                } else {
                    badgeEl.style.background = "var(--danger)";
                    badgeEl.textContent = "Pathology";
                }
            }
        }

        // Render AI Multi-Agent Swarm Finding
        if (data.ai_swarm) {
            const findingEl = document.getElementById('ai-swarm-finding');
            const pupilEl = document.getElementById('ai-pupil-score');
            const vascEl = document.getElementById('ai-vasc-score');
            const aiBadgeEl = document.getElementById('ai-swarm-badge');

            if (findingEl) findingEl.textContent = data.ai_swarm.swarm_finding || "Healthy Anatomy";
            if (pupilEl) pupilEl.textContent = Number(data.ai_swarm.pupil_opacity || 0).toFixed(1);
            if (vascEl) vascEl.textContent = `${Number(data.ai_swarm.vascular_density || 0).toFixed(1)}%`;

            if (aiBadgeEl) {
                const finding = (data.ai_swarm.swarm_finding || "").toLowerCase();
                if (finding.includes('healthy') || finding.includes('normal') || finding.includes('clear')) {
                    aiBadgeEl.style.background = "var(--success)";
                    aiBadgeEl.textContent = "Normal";
                } else {
                    aiBadgeEl.style.background = "var(--danger)";
                    aiBadgeEl.textContent = "Flagged";
                }
            }
        }

        // Render Merged Clinical Consensus
        if (data.merged_consensus) {
            const mergedTitleEl = document.getElementById('merged-verdict-title');
            const mergedSummaryEl = document.getElementById('merged-verdict-summary');
            const mergedBadgeEl = document.getElementById('merged-verdict-badge');

            if (mergedTitleEl) mergedTitleEl.textContent = data.merged_consensus.final_verdict || data.title || "No Disease (Healthy)";
            if (mergedSummaryEl) mergedSummaryEl.textContent = data.merged_consensus.consensus_summary || "Multi-agent cross-verification complete.";

            if (mergedBadgeEl) {
                if (data.category === 'NORMAL') {
                    mergedBadgeEl.style.background = "var(--success)";
                    mergedBadgeEl.textContent = "Healthy";
                } else if (data.category === 'RETAKE' || data.category === 'NOT_EYE') {
                    mergedBadgeEl.style.background = "#d97706";
                    mergedBadgeEl.textContent = "Review";
                } else {
                    mergedBadgeEl.style.background = "var(--danger)";
                    mergedBadgeEl.textContent = "Action Req";
                }
            }
        }
        
        // Render Oculomics Tri-Biomarker Agents (Liver, Blood, Cardio)
        if (data.oculomics) {
            // 1. Liver Specialist
            const liver = data.oculomics.liver;
            if (liver && typeof liver === 'object') {
                const syiEl = document.getElementById('oculomics-liver-syi');
                const riskEl = document.getElementById('oculomics-liver-risk');
                const badgeEl = document.getElementById('oculomics-liver-badge');
                const descEl = document.getElementById('oculomics-liver-desc');
                
                if (syiEl) syiEl.textContent = liver.yellow_index !== undefined ? Number(liver.yellow_index).toFixed(2) : "1.00";
                if (riskEl) riskEl.textContent = `${liver.jaundice_risk !== undefined ? Number(liver.jaundice_risk).toFixed(1) : "0.0"}%`;
                if (descEl) descEl.textContent = liver.status || "Healthy Scleral Chromaticity";
                
                if (badgeEl) {
                    if (liver.level === 'HIGH' || (liver.jaundice_risk && liver.jaundice_risk > 60)) {
                        badgeEl.style.background = "#fee2e2";
                        badgeEl.style.color = "#991b1b";
                        badgeEl.textContent = "ELEVATED RISK";
                    } else if (liver.level === 'MODERATE' || (liver.jaundice_risk && liver.jaundice_risk > 30)) {
                        badgeEl.style.background = "#fef3c7";
                        badgeEl.style.color = "#92400e";
                        badgeEl.textContent = "BORDERLINE";
                    } else {
                        badgeEl.style.background = "#fef9c3";
                        badgeEl.style.color = "#a16207";
                        badgeEl.textContent = "NORMAL";
                    }
                }
            } else if (document.getElementById('oculomics-liver-desc')) {
                document.getElementById('oculomics-liver-desc').textContent = data.oculomics.liver || "N/A";
            }

            // 2. Blood Specialist
            const blood = data.oculomics.blood;
            if (blood && typeof blood === 'object') {
                const erythemaEl = document.getElementById('oculomics-blood-erythema');
                const pallorEl = document.getElementById('oculomics-blood-pallor');
                const riskEl = document.getElementById('oculomics-blood-risk');
                const badgeEl = document.getElementById('oculomics-blood-badge');
                const descEl = document.getElementById('oculomics-blood-desc');
                
                if (erythemaEl) erythemaEl.textContent = blood.erythema_ratio !== undefined ? Number(blood.erythema_ratio).toFixed(2) : "1.40";
                if (pallorEl) pallorEl.textContent = blood.pallor_score !== undefined ? Number(blood.pallor_score).toFixed(1) : "0.0";
                if (riskEl) riskEl.textContent = `${blood.anemia_risk !== undefined ? Number(blood.anemia_risk).toFixed(1) : "0.0"}%`;
                if (descEl) descEl.textContent = blood.status || "Healthy Tissue Perfusion";
                
                if (badgeEl) {
                    if (blood.level === 'HIGH' || (blood.anemia_risk && blood.anemia_risk > 60)) {
                        badgeEl.style.background = "#fee2e2";
                        badgeEl.style.color = "#991b1b";
                        badgeEl.textContent = "ANEMIA RISK";
                    } else if (blood.level === 'MODERATE' || (blood.anemia_risk && blood.anemia_risk > 30)) {
                        badgeEl.style.background = "#ffedd5";
                        badgeEl.style.color = "#c2410c";
                        badgeEl.textContent = "MILD PALLOR";
                    } else {
                        badgeEl.style.background = "#ffe4e6";
                        badgeEl.style.color = "#be123c";
                        badgeEl.textContent = "NORMAL";
                    }
                }
            } else if (document.getElementById('oculomics-blood-desc')) {
                document.getElementById('oculomics-blood-desc').textContent = data.oculomics.blood || "N/A";
            }

            // 3. Cardiac Specialist
            const cardio = data.oculomics.cardio;
            if (cardio && typeof cardio === 'object') {
                const densityEl = document.getElementById('oculomics-cardio-density');
                const riskEl = document.getElementById('oculomics-cardio-risk');
                const badgeEl = document.getElementById('oculomics-cardio-badge');
                const descEl = document.getElementById('oculomics-cardio-desc');
                
                if (densityEl) densityEl.textContent = `${cardio.vascular_density !== undefined ? Number(cardio.vascular_density).toFixed(2) : "0.00"}%`;
                if (riskEl) riskEl.textContent = `${cardio.hypertension_risk !== undefined ? Number(cardio.hypertension_risk).toFixed(1) : "0.0"}%`;
                if (descEl) descEl.textContent = cardio.status || "Normal Micro-Vascular Baseline";
                
                if (badgeEl) {
                    if (cardio.level === 'HIGH' || (cardio.hypertension_risk && cardio.hypertension_risk > 65)) {
                        badgeEl.style.background = "#fee2e2";
                        badgeEl.style.color = "#991b1b";
                        badgeEl.textContent = "HYPERTENSION RISK";
                    } else if (cardio.level === 'MODERATE' || (cardio.hypertension_risk && cardio.hypertension_risk > 30)) {
                        badgeEl.style.background = "#ffedd5";
                        badgeEl.style.color = "#c2410c";
                        badgeEl.textContent = "MODERATE";
                    } else {
                        badgeEl.style.background = "#ffedd5";
                        badgeEl.style.color = "#c2410c";
                        badgeEl.textContent = "NORMAL";
                    }
                }
            } else if (document.getElementById('oculomics-cardio-desc')) {
                document.getElementById('oculomics-cardio-desc').textContent = data.oculomics.cardio || "N/A";
            }
        }
        
        const zkpEl = document.getElementById('zkp-hash');
        if (zkpEl) zkpEl.textContent = data.zkp_hash || "N/A";

        // Theming based on category
        resultCard.className = 'result-banner'; 
        
        if (data.category === 'NORMAL') {
            resultCard.classList.add('status-normal');
            resultIcon.textContent = '✅';
        } else if (data.category === 'RETAKE' || data.category === 'NOT_EYE') {
             resultCard.classList.add('status-warning');
             resultIcon.textContent = '⚠️';
             resultConfidence.textContent = 'N/A';
        } else {
            resultCard.classList.add('status-alert');
            resultIcon.textContent = '⚠️';
        }

        if (data.category !== 'RETAKE' && data.category !== 'NOT_EYE') {
             saveToHistory(data);
        }
    }

    btnReset.addEventListener('click', resetUI);
});
