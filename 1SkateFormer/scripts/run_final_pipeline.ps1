# run_final_pipeline.ps1
# =====================
# End-to-end pseudo-label pipeline for SkateFormer.
# Runs all 4 protocols sequentially:
#   NTU60-CS, NTU60-CV, NTU120-CSub, NTU120-CSet
#
# Prerequisites:
#   - Python environment with all dependencies installed.
#   - Pretrained SkateFormer checkpoint for each protocol
#     (set CKPT_NTU60_CS etc. below, or pass as arguments).
#   - Data files in place (see config/ for paths).
#
# Usage:
#   .\scripts\run_final_pipeline.ps1
#   .\scripts\run_final_pipeline.ps1 -Protocol ntu60_cs -Phase all
#   .\scripts\run_final_pipeline.ps1 -Protocol ntu60_cs -Phase oof_only
#
# Phases:
#   all          Run full pipeline (template → train → OOF → refine → student → audit)
#   template     Build template labels only
#   train        Train fixed-template decoder only
#   oof          Run OOF predictions only
#   refine       Run label refinement only
#   student      Train final student on refined labels
#   eval         Evaluate concept metrics
#   audit        Select audit samples
#   oof_only     template + oof (no student training)

param(
    [string]$Protocol  = "all",   # "all" | "ntu60_cs" | "ntu60_cv" | "ntu120_csub" | "ntu120_cset"
    [string]$Phase     = "all",   # see above
    [string]$Device    = "0",
    [int]$Seed         = 1,
    [int]$OofEpochs    = 20,
    [int]$StudentEpochs= 10,
    [int]$BatchSize    = 64,
    [int]$NumWorker    = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---- Checkpoints: set these to your pretrained SkateFormer checkpoints ----
$CKPT = @{
    "ntu60_cs"    = ""   # e.g. "work_dir/ntu/cs/SkateFormer_j/runs-450-xxxxx.pt"
    "ntu60_cv"    = ""
    "ntu120_csub" = ""
    "ntu120_cset" = ""
}

# ---- Protocol definitions ----
$PROTOCOLS = @{
    "ntu60_cs" = @{
        Config      = "config/final/ntu60_cs.yaml"
        DataTrain   = "data/ntu/NTU60_CS.npz"
        DataTest    = "data/ntu/NTU60_CS.npz"
        NumClasses  = 60
        WorkDir     = "work_dir/final_pipeline/ntu60_cs"
    }
    "ntu60_cv" = @{
        Config      = "config/final/ntu60_cv.yaml"
        DataTrain   = "data/ntu/NTU60_CV.npz"
        DataTest    = "data/ntu/NTU60_CV.npz"
        NumClasses  = 60
        WorkDir     = "work_dir/final_pipeline/ntu60_cv"
    }
    "ntu120_csub" = @{
        Config      = "config/final/ntu120_csub.yaml"
        DataTrain   = "data/ntu120/NTU120_CSub.npz"
        DataTest    = "data/ntu120/NTU120_CSub.npz"
        NumClasses  = 120
        WorkDir     = "work_dir/final_pipeline/ntu120_csub"
    }
    "ntu120_cset" = @{
        Config      = "config/final/ntu120_cset.yaml"
        DataTrain   = "data/ntu120/NTU120_CSet.npz"
        DataTest    = "data/ntu120/NTU120_CSet.npz"
        NumClasses  = 120
        WorkDir     = "work_dir/final_pipeline/ntu120_cset"
    }
}

# ---- Determine which protocols to run ----
if ($Protocol -eq "all") {
    $ProtoList = @("ntu60_cs", "ntu60_cv", "ntu120_csub", "ntu120_cset")
} else {
    $ProtoList = @($Protocol)
}

# ---- Helper functions ----
function Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $msg" -ForegroundColor Cyan
}

function Run-Command {
    param([string]$Cmd)
    Log "Running: $Cmd"
    Invoke-Expression $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Command failed with exit code $LASTEXITCODE: $Cmd"
        exit $LASTEXITCODE
    }
}

function Run-Protocol {
    param([string]$proto)

    $p = $PROTOCOLS[$proto]
    $work   = $p.WorkDir
    $config = $p.Config
    $nc     = $p.NumClasses
    $ckpt   = $CKPT[$proto]

    $pseudoDir  = "$work/pseudo_labels"
    $metricsDir = "$work/metrics"
    $auditDir   = "$work/audit"
    $oofDir     = "$work/oof"

    New-Item -ItemType Directory -Force -Path $pseudoDir  | Out-Null
    New-Item -ItemType Directory -Force -Path $metricsDir | Out-Null
    New-Item -ItemType Directory -Force -Path $auditDir   | Out-Null
    New-Item -ItemType Directory -Force -Path $oofDir     | Out-Null

    # ---------------------------------------------------------------
    # Step 1: Build template labels
    # ---------------------------------------------------------------
    if ($Phase -in @("all", "template", "oof_only")) {
        Log "[$proto] Step 1: Building template labels..."

        Run-Command "python tools/build_template_labels.py ``
            --data_path $($p.DataTrain) --split train ``
            --num_classes $nc --out_dir $pseudoDir --protocol $proto"

        Run-Command "python tools/build_template_labels.py ``
            --data_path $($p.DataTest) --split test ``
            --num_classes $nc --out_dir $pseudoDir --protocol $proto"
    }

    # ---------------------------------------------------------------
    # Step 2: Train fixed-template decoder (Phase 2 baseline)
    # ---------------------------------------------------------------
    if ($Phase -in @("all", "train")) {
        Log "[$proto] Step 2: Training fixed-template decoder..."

        $extraArgs = ""
        if ($ckpt -ne "") { $extraArgs = "--backbone_ckpt $ckpt" }

        Run-Command "python main_concept.py ``
            --config $config ``
            --work_dir $work/baseline ``
            --phase train_concept ``
            --seed $Seed ``
            --device $Device ``
            --batch_size $BatchSize ``
            --num_worker $NumWorker ``
            $extraArgs"
    }

    # ---------------------------------------------------------------
    # Step 3: OOF predictions (teacher)
    # ---------------------------------------------------------------
    if ($Phase -in @("all", "oof", "oof_only")) {
        Log "[$proto] Step 3: Running 3-fold OOF predictions..."

        $extraArgs = ""
        if ($ckpt -ne "") { $extraArgs = "--backbone_ckpt $ckpt" }  # passed via config

        Run-Command "python tools/make_oof_predictions.py ``
            --config $config ``
            --out_dir $pseudoDir ``
            --n_folds 3 ``
            --num_epoch $OofEpochs ``
            --batch_size $BatchSize ``
            --num_worker $NumWorker ``
            --seed $Seed ``
            --device $Device ``
            --early_stop 5"
    }

    # ---------------------------------------------------------------
    # Step 4: Evaluate OOF concept quality
    # ---------------------------------------------------------------
    if ($Phase -in @("all", "eval")) {
        Log "[$proto] Step 4: Evaluating OOF concept predictions..."

        if (Test-Path "$pseudoDir/oof_probs.npz") {
            Run-Command "python tools/evaluate_concepts.py ``
                --oof_npz $pseudoDir/oof_probs.npz ``
                --template_npz $pseudoDir/template_labels_train.npz ``
                --concept_csv concepts/ntu_rgbd_60_120.csv ``
                --num_classes $nc ``
                --out $metricsDir/oof_concept_eval.json"
        } else {
            Log "[$proto] WARNING: oof_probs.npz not found, skipping OOF eval"
        }
    }

    # ---------------------------------------------------------------
    # Step 5: Refine pseudo-labels
    # ---------------------------------------------------------------
    if ($Phase -in @("all", "refine")) {
        Log "[$proto] Step 5: Refining pseudo-labels (round 1)..."

        if (Test-Path "$pseudoDir/oof_probs.npz") {
            Run-Command "python tools/refine_pseudolabels.py ``
                --template_npz $pseudoDir/template_labels_train.npz ``
                --oof_npz $pseudoDir/oof_probs.npz ``
                --out_dir $pseudoDir ``
                --round 1 ``
                --pos_threshold 0.6 ``
                --flip_threshold 0.25 ``
                --new_pos_threshold 0.80 ``
                --min_confidence 0.50"
        } else {
            Log "[$proto] WARNING: oof_probs.npz not found, skipping refinement"
        }
    }

    # ---------------------------------------------------------------
    # Step 6: Select audit samples
    # ---------------------------------------------------------------
    if ($Phase -in @("all", "audit")) {
        Log "[$proto] Step 6: Selecting audit samples..."

        if (Test-Path "$pseudoDir/oof_probs.npz") {
            $refinedArg = ""
            if (Test-Path "$pseudoDir/refined_labels_v1.npz") {
                $refinedArg = "--refined_npz $pseudoDir/refined_labels_v1.npz"
            }
            Run-Command "python tools/select_audit_samples.py ``
                --oof_npz $pseudoDir/oof_probs.npz ``
                --template_npz $pseudoDir/template_labels_train.npz ``
                --out_dir $auditDir ``
                $refinedArg ``
                --n_disagree 75 --n_new_pos 75 --n_low_conf 50 --n_random 50 ``
                --seed 42"
        }
    }

    # ---------------------------------------------------------------
    # Step 7: Train final student on refined labels
    # ---------------------------------------------------------------
    if ($Phase -in @("all", "student")) {
        Log "[$proto] Step 7: Training final student (refined labels)..."

        $refinedNpz = "$pseudoDir/refined_labels_v1.npz"
        if (-not (Test-Path $refinedNpz)) {
            $refinedNpz = "$pseudoDir/template_labels_train.npz"
            Log "[$proto] WARNING: refined_labels_v1.npz not found, using template labels"
        }

        $extraArgs = ""
        if ($ckpt -ne "") { $extraArgs = "--backbone_ckpt $ckpt" }

        Run-Command "python main_concept.py ``
            --config $config ``
            --work_dir $work/student ``
            --phase train_concept ``
            --seed $Seed ``
            --device $Device ``
            --batch_size $BatchSize ``
            --num_worker $NumWorker ``
            --num_epoch $StudentEpochs ``
            --loss_type soft_bce ``
            --train_feeder_args.pseudo_label_npz $refinedNpz ``
            $extraArgs"
    }

    Log "[$proto] Protocol complete!"
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
Log "Starting SkateFormer concept pipeline"
Log "Protocols: $($ProtoList -join ', ')"
Log "Phase: $Phase"

foreach ($proto in $ProtoList) {
    Log "=========================================="
    Log "Protocol: $proto"
    Log "=========================================="
    Run-Protocol -proto $proto
}

Log "All done!"
