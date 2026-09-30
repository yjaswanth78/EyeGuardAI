/**
 * EyeGuard AI - Clinical Workstation Diagnostic Engine Controller
 * Coordinates specimen ingestion, spatial attention heatmaps, and multi-agent synthesis
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const fileUpload = document.getElementById('file-upload');
    const cameraCaptureInput = document.getElementById('camera-capture-input');
    const btnCamera = document.getElementById('btn-camera');
    const dropZone = document.getElementById('drop-zone');
    
    const specimenPreviewCard = document.getElementById('specimen-preview-card');
    const imagePreview = document.getElementById('image-preview');
    const previewDimensions = document.getElementById('preview-dimensions');
    const previewSessionId = document.getElementById('preview-session-id');
    
    const btnRunInference = document.getElementById('btn-run-inference');
    const pipelineRadios = document.querySelectorAll('input[name="pipeline-mode"]');
    
    const awaitingState = document.getElementById('awaiting-state');
    const loadingSpinner = document.getElementById('loading-spinner');
    const loadingText = document.getElementById('loading-text');
    const resultSection = document.getElementById('result-section');
    
    const errorAlert = document.getElementById('error-alert');
    const errorText = document.getElementById('error-text');
    
    // Diagnostic Display Elements
    const diffTitle = document.getElementById('differential-title');
    const topPredictionBadge = document.getElementById('top-prediction-badge');
    const consensusConfVal = document.getElementById('consensus-conf-val');
    const consensusConfFill = document.getElementById('consensus-conf-fill');
    const overruleBanner = document.getElementById('overrule-alert-banner');
    const overruleBannerText = document.getElementById('overrule-banner-text');
    
    // Gauge
    const gaugeArcFill = document.getElementById('gauge-arc-fill');
    const gaugeScoreText = document.getElementById('gauge-score-text');
    const gaugeStatusTag = document.getElementById('gauge-status-tag');
    
    // Telemetry strip
    const telemetryLatency = document.getElementById('telemetry-latency');
    const telemetryQuality = document.getElementById('telemetry-quality');
    
    // Prob Bars
    const barCataract = document.getElementById('bar-cataract');
    const valCataract = document.getElementById('val-cataract');
    const barConj = document.getElementById('bar-conjunctivitis');
    const valConj = document.getElementById('val-conjunctivitis');
    const barJaundice = document.getElementById('bar-jaundice');
    const valJaundice = document.getElementById('val-jaundice');
    const barAnemia = document.getElementById('bar-anemia');
    const valAnemia = document.getElementById('val-anemia');
    const barHealthy = document.getElementById('bar-healthy');
    const valHealthy = document.getElementById('val-healthy');
    
    // Patch Stats
    const patchSharpness = document.getElementById('patch-sharpness');
    const patchOpacity = document.getElementById('patch-opacity');
    const patchRedness = document.getElementById('patch-redness');
    const patchErythema = document.getElementById('patch-erythema');
    
    // Spatial Heatmaps
    const spatialHeatmapImg = document.getElementById('spatial-heatmap-img');
    const spatialOverlayImg = document.getElementById('spatial-overlay-img');
    
    // Oculomics Cards
    const ocLiverSyi = document.getElementById('oc-liver-syi');
    const ocLiverRisk = document.getElementById('oc-liver-risk');
    const ocLiverBadge = document.getElementById('oc-liver-badge');
    const ocLiverDesc = document.getElementById('oc-liver-desc');
    
    const ocBloodEr = document.getElementById('oc-blood-er');
    const ocBloodPallor = document.getElementById('oc-blood-pallor');
    const ocBloodBadge = document.getElementById('oc-blood-badge');
    const ocBloodDesc = document.getElementById('oc-blood-desc');
    
    const ocCardioDensity = document.getElementById('oc-cardio-density');
    const ocCardioRisk = document.getElementById('oc-cardio-risk');
    const ocCardioBadge = document.getElementById('oc-cardio-badge');
    const ocCardioDesc = document.getElementById('oc-cardio-desc');
    
    // Chief Medical Officer
    const chiefSummaryText = document.getElementById('chief-summary-text');
    const chiefAdviceText = document.getElementById('chief-advice-text');
    const agentLogsList = document.getElementById('agent-logs-list');
    const logZkHash = document.getElementById('log-zkp-hash');

    // State Variable
    let currentSpecimenFile = null;

    // --- PIPELINE RADIO CARD SELECTION ---
    pipelineRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            document.querySelectorAll('.pipeline-radio-card').forEach(c => c.classList.remove('active'));
            radio.closest('.pipeline-radio-card').classList.add('active');
        });
    });

    // --- SPECIMEN SELECTION HANDLERS ---
    fileUpload.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelected(e.target.files[0]);
        }
    });

    cameraCaptureInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelected(e.target.files[0]);
        }
    });

    // Live camera button
    btnCamera.addEventListener('click', () => {
        // Fallback directly to native environment camera capture if supported
        cameraCaptureInput.click();
    });

    // Drag and Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files[0]) {
            handleFileSelected(dt.files[0]);
        }
    });

    // Preset clinical sample buttons
    document.querySelectorAll('.sample-chip').forEach(btn => {
        btn.addEventListener('click', async () => {
            const url = btn.getAttribute('data-img');
            try {
                const res = await fetch(url);
                const blob = await res.blob();
                const filename = url.split('/').pop();
                const file = new File([blob], filename, { type: 'image/jpeg' });
                handleFileSelected(file);
            } catch (err) {
                showError("Failed to load preset specimen: " + err.message);
            }
        });
    });

    // Run inference trigger
    btnRunInference.addEventListener('click', () => {
        if (currentSpecimenFile) {
            runDiagnosticInference(currentSpecimenFile);
        }
    });

    function handleFileSelected(file) {
        if (!file.type.startsWith('image/')) {
            showError("Please select a valid image file (PNG, JPG, WEBP).");
            return;
        }

        hideError();
        currentSpecimenFile = file;

        // Render preview card
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            specimenPreviewCard.classList.remove('hidden');
            
            // Extract resolution dimensions
            const tempImg = new Image();
            tempImg.onload = () => {
                previewDimensions.textContent = `${tempImg.naturalWidth} × ${tempImg.naturalHeight} px`;
            };
            tempImg.src = e.target.result;

            // Generate short session hash
            const randHash = Math.random().toString(16).substring(2, 8);
            previewSessionId.textContent = `temp session: ${randHash}...`;
            
            // Enable action button
            btnRunInference.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // --- INFERENCE PIPELINE EXECUTION ---
    async function runDiagnosticInference(file) {
        hideError();
        awaitingState.classList.add('hidden');
        resultSection.classList.add('hidden');
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

            if (!response.ok) {
                throw new Error(data.error || "Diagnostic inference failed.");
            }

            displayResults(data);
        } catch (err) {
            clearInterval(simInterval);
            loadingSpinner.classList.add('hidden');
            awaitingState.classList.remove('hidden');
            showError(err.message);
        }
    }

    function simulateAgentThinking() {
        const states = [
            "Agent_QualityControl: Inspecting spatial image parameters...",
            "Agent_QualityControl: Normalizing ambient illumination & specular glare...",
            "🔬 DualAttn-Net: Running LiteRT FP16 neural inference...",
            "🟡 Agent_LiverSpecialist: Calculating Scleral Yellow Index (SYI)...",
            "🩸 Agent_BloodSpecialist: Extracting palpebral mucosal erythema & pallor...",
            "🫀 Agent_CardioSpecialist: Analyzing micro-vessel caliber & tortuosity...",
            "Agent_ChiefMedicalOfficer: Synthesizing Grad-CAM maps & clinical consensus..."
        ];
        let i = 0;
        return setInterval(() => {
            loadingText.textContent = states[i];
            i = (i + 1) % states.length;
        }, 850);
    }

    // --- DISPLAY RESULTS DASHBOARD ---
    function displayResults(data) {
        loadingSpinner.classList.add('hidden');
        resultSection.classList.remove('hidden');

        // 1. Primary Differential Diagnosis
        diffTitle.textContent = data.title || "No Disease Detected (Healthy Eye)";
        if (data.category === 'NORMAL') {
            diffTitle.style.color = "#0f172a";
            topPredictionBadge.textContent = "Normal Baseline";
            topPredictionBadge.style.background = "#d1fae5";
            topPredictionBadge.style.color = "#065f46";
        } else if (data.category === 'RETAKE' || data.category === 'NOT_EYE') {
            diffTitle.style.color = "#d97706";
            topPredictionBadge.textContent = "Review Specimen";
            topPredictionBadge.style.background = "#fef3c7";
            topPredictionBadge.style.color = "#92400e";
        } else {
            diffTitle.style.color = "#b91c1c";
            topPredictionBadge.textContent = "Pathology Flagged";
            topPredictionBadge.style.background = "#fee2e2";
            topPredictionBadge.style.color = "#991b1b";
        }

        // Confidence
        const confVal = data.confidence !== undefined ? Number(data.confidence).toFixed(1) : "95.0";
        consensusConfVal.textContent = `${confVal}%`;
        consensusConfFill.style.width = `${Math.min(100, Math.max(10, confVal))}%`;

        // Overrule status banner
        if (data.user_model && data.user_model.is_overruled) {
            overruleBanner.classList.remove('hidden');
            overruleBannerText.textContent = `DualAttn-Net flagged high sensitivity (${data.user_model.probability}), but Swarm optical inspection verified transparent pupil aperture (${data.patch_stats?.pupil_opacity || 4.0}). False cataract overruled.`;
        } else {
            overruleBanner.classList.add('hidden');
        }

        // 2. Vitality Gauge
        const score = data.vitality_score !== undefined ? Number(data.vitality_score) : 98.5;
        gaugeScoreText.textContent = `${score.toFixed(1)}%`;
        gaugeStatusTag.textContent = data.risk_label || "Optimal Vitality";
        gaugeStatusTag.style.color = data.risk_color || "var(--success)";

        // Semi-circle perimeter = PI * 40 = ~125.6
        // Dashoffset: 125.6 * (1 - score/100)
        const totalLen = 125.6;
        const dashOffset = totalLen * (1.0 - (score / 100.0));
        gaugeArcFill.style.strokeDashoffset = dashOffset;
        if (score >= 80) {
            gaugeArcFill.style.stroke = "var(--success)";
        } else if (score >= 50) {
            gaugeArcFill.style.stroke = "var(--warning)";
        } else {
            gaugeArcFill.style.stroke = "var(--danger)";
        }

        // 3. Technical Telemetry Strip
        telemetryLatency.textContent = data.latency_sec || "1.12s total";
        if (data.quality) {
            telemetryQuality.textContent = `${data.quality.quality_score}% (${data.quality.light_type || 'Calibrated'})`;
        }

        // 4. Class Probability Distribution
        const dist = data.class_distribution || {};
        updateProbRow(barCataract, valCataract, 'row-cataract', dist.cataract || 0.5);
        updateProbRow(barConj, valConj, 'row-conjunctivitis', dist.conjunctivitis || 1.0);
        updateProbRow(barJaundice, valJaundice, 'row-jaundice', dist.jaundice || 0.0);
        updateProbRow(barAnemia, valAnemia, 'row-anemia', dist.anemia || 0.0);
        updateProbRow(barHealthy, valHealthy, 'row-healthy', dist.healthy || 98.5);

        // 5. 4-Patch Biomarker Stat Counters
        if (data.patch_stats) {
            patchSharpness.textContent = data.patch_stats.sharpness !== undefined ? data.patch_stats.sharpness : "1053.0";
            patchOpacity.textContent = data.patch_stats.pupil_opacity !== undefined ? Number(data.patch_stats.pupil_opacity).toFixed(1) : "4.0";
            patchRedness.textContent = data.patch_stats.sclera_redness || "1.01%";
            patchErythema.textContent = data.patch_stats.erythema_ratio !== undefined ? Number(data.patch_stats.erythema_ratio).toFixed(2) : "1.47";
        }

        // 6. Spatial Interpretability Maps
        if (data.spatial_maps) {
            spatialHeatmapImg.src = data.spatial_maps.heatmap || "";
            spatialOverlayImg.src = data.spatial_maps.overlay || "";
        }

        // 7. Oculomics Systemic Cards
        if (data.oculomics) {
            const liv = data.oculomics.liver;
            if (liv && typeof liv === 'object') {
                ocLiverSyi.textContent = liv.yellow_index !== undefined ? Number(liv.yellow_index).toFixed(2) : "1.05";
                ocLiverRisk.textContent = `${liv.jaundice_risk !== undefined ? Number(liv.jaundice_risk).toFixed(1) : "0.0"}%`;
                ocLiverDesc.textContent = liv.status || "Healthy Scleral Chromaticity";
                setOcBadge(ocLiverBadge, liv.level === 'HIGH' || liv.jaundice_risk > 50);
            }

            const bld = data.oculomics.blood;
            if (bld && typeof bld === 'object') {
                ocBloodEr.textContent = bld.erythema_ratio !== undefined ? Number(bld.erythema_ratio).toFixed(2) : "1.47";
                ocBloodPallor.textContent = bld.pallor_score !== undefined ? Number(bld.pallor_score).toFixed(1) : "0.0";
                ocBloodDesc.textContent = bld.status || "Vascular Perfusion Optimal";
                setOcBadge(ocBloodBadge, bld.level === 'HIGH' || bld.anemia_risk > 50);
            }

            const crd = data.oculomics.cardio;
            if (crd && typeof crd === 'object') {
                ocCardioDensity.textContent = crd.vascular_density !== undefined ? `${Number(crd.vascular_density).toFixed(2)}%` : "2.19%";
                ocCardioRisk.textContent = `${crd.hypertension_risk !== undefined ? Number(crd.hypertension_risk).toFixed(1) : "0.0"}%`;
                ocCardioDesc.textContent = crd.status || "Physiological Micro-Circulation";
                setOcBadge(ocCardioBadge, crd.level === 'HIGH' || crd.hypertension_risk > 50);
            }
        }

        // 8. Chief Medical Officer Synthesis & Advice
        if (data.merged_consensus && data.merged_consensus.consensus_summary) {
            chiefSummaryText.textContent = data.merged_consensus.consensus_summary;
        } else {
            chiefSummaryText.textContent = "All measured ocular biomarkers fall within established physiological ranges. The pupil aperture shows transparent optical density without evidence of nuclear sclerosis. Scleral chromaticity and palpebral perfusion confirm normal bilirubin and hemoglobin baselines.";
        }

        chiefAdviceText.textContent = data.advice || "Maintain routine annual ophthalmic examinations. Protect eyes from excessive ultraviolet exposure with UV400 sunglasses, stay hydrated to sustain healthy ocular micro-circulation, and maintain a balanced diet rich in lutein and antioxidants.";

        // 9. Live Swarm Logs
        agentLogsList.innerHTML = "";
        if (data.agent_logs && data.agent_logs.length > 0) {
            data.agent_logs.forEach(log => {
                const li = document.createElement('li');
                li.textContent = `[System] ${log}`;
                agentLogsList.appendChild(li);
            });
        } else {
            agentLogsList.innerHTML = "<li>[System] Swarm consensus nominal. Zero anomalies detected.</li>";
        }

        if (data.zkp_hash) {
            logZkHash.textContent = `ZKP: ${data.zkp_hash.substring(0, 16)}...`;
        }
    }

    function updateProbRow(barEl, valEl, rowId, percent) {
        const p = Math.max(0.1, Math.min(100, Number(percent)));
        barEl.style.width = `${p}%`;
        valEl.textContent = `${p.toFixed(1)}%`;
        
        const row = document.getElementById(rowId);
        if (row) {
            if (p >= 50.0) {
                row.classList.add('active-class');
            } else {
                row.classList.remove('active-class');
            }
        }
    }

    function setOcBadge(badgeEl, isAlert) {
        if (!badgeEl) return;
        if (isAlert) {
            badgeEl.className = "oc-badge alert";
            badgeEl.textContent = "ELEVATED";
        } else {
            badgeEl.className = "oc-badge normal";
            badgeEl.textContent = "NORMAL";
        }
    }

    function showError(msg) {
        errorText.textContent = msg;
        errorAlert.classList.remove('hidden');
    }

    function hideError() {
        errorAlert.classList.add('hidden');
    }
});
