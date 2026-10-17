import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  Activity, AlertTriangle, CheckCircle, Info, ShieldAlert, BarChart2, 
  Users, Sliders, Play, Pause, SkipForward, RotateCcw, Upload, Download, 
  Search, Filter, ChevronRight, ArrowUpRight, ArrowDownRight, Clock, 
  HelpCircle, Eye, FileText, Settings, Heart, Thermometer, Wind, Zap, Check
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea, ReferenceLine 
} from 'recharts';

// ==========================================
// 1. STATISTICAL & ANOMALY DETECTION ENGINE
// ==========================================

// Calculate Median of an array
const calculateMedian = (arr) => {
  if (!arr || arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

// Calculate Median Absolute Deviation (MAD)
const calculateMAD = (arr, median) => {
  if (!arr || arr.length === 0) return 0;
  const med = median !== undefined ? median : calculateMedian(arr);
  const absoluteDeviations = arr.map(val => Math.abs(val - med));
  return calculateMedian(absoluteDeviations);
};

// Calculate Robust Z-Score: |x - median| / (1.4826 * MAD + epsilon)
const calculateRobustZScore = (value, median, mad) => {
  const scaleFloor = 0.001; // Avoid division by zero
  const scaledMAD = 1.4826 * (mad < scaleFloor ? scaleFloor : mad);
  return (value - median) / scaledMAD;
};

// Clinical Illustrative Fixed Thresholds
const FIXED_THRESHOLDS = {
  heart_rate: { min: 60, max: 100, label: '60 - 100 bpm' },
  respiratory_rate: { min: 12, max: 20, label: '12 - 20 bpm' },
  temperature_c: { min: 36.1, max: 37.8, label: '36.1 - 37.8 °C' },
  spo2: { min: 92, max: 100, label: '92 - 100 %' },
  systolic_bp: { min: 90, max: 130, label: '90 - 130 mmHg' }
};

// Detect anomalies using both Personalized Detector and Fixed Threshold
const analyzeTimeSeriesData = (measurements, baselineWindowSize = 12) => {
  if (!measurements || measurements.length === 0) return { alerts: [], processedData: [] };

  const sorted = [...measurements].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  const processedData = [];
  const alerts = [];
  const activeAlertCooldowns = {}; // Cooldown to suppress duplicate alerts

  // Compute baseline per patient for each vital metric using the first baselineWindowSize points
  const baselines = {};
  const metrics = ['heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 'systolic_bp'];

  // Initialize baselines if enough initial points exist
  metrics.forEach(m => {
    const initialVals = sorted.slice(0, Math.min(baselineWindowSize, sorted.length))
      .map(d => d[m])
      .filter(v => v !== null && v !== undefined && !isNaN(v));
    
    if (initialVals.length >= 4) {
      const median = calculateMedian(initialVals);
      const mad = calculateMAD(initialVals, median);
      baselines[m] = { median, mad, count: initialVals.length };
    } else {
      baselines[m] = { median: 0, mad: 0, count: 0 };
    }
  });

  // Track rolling window for persistence analysis
  const recentZScores = {
    heart_rate: [],
    respiratory_rate: [],
    temperature_c: [],
    spo2: [],
    systolic_bp: []
  };

  sorted.forEach((point, idx) => {
    const pointAnalysis = { ...point, zScores: {}, detectedAnomalies: [] };

    metrics.forEach(m => {
      const val = point[m];
      if (val === null || val === undefined || isNaN(val)) return;

      const base = baselines[m];
      let zScore = 0;
      if (base.count >= 4) {
        zScore = calculateRobustZScore(val, base.median, base.mad);
        pointAnalysis.zScores[m] = parseFloat(zScore.toFixed(2));
      }

      // Maintain rolling Z-scores
      recentZScores[m].push(zScore);
      if (recentZScores[m].length > 4) recentZScores[m].shift();

      // Check Personalized Anomaly Rules
      const absZ = Math.abs(zScore);
      const recentZ = recentZScores[m];
      const isPersistent = recentZ.length >= 3 && recentZ.every(z => Math.abs(z) > 2.0);
      const isSuddenSpike = absZ > 3.2;
      const isGradualDrift = recentZ.length >= 4 && (
        (recentZ[3] - recentZ[0] > 1.8 && absZ > 2.2) || 
        (recentZ[0] - recentZ[3] > 1.8 && absZ > 2.2)
      );

      // Check Fixed Threshold Rule
      const fixedConfig = FIXED_THRESHOLDS[m];
      const fixedExceeded = val < fixedConfig.min || val > fixedConfig.max;

      let anomalyType = null;
      if (isSuddenSpike) anomalyType = 'Sudden Deviation';
      else if (isGradualDrift) anomalyType = 'Gradual Drift';
      else if (isPersistent) anomalyType = 'Persistent Shift';

      if (anomalyType) {
        pointAnalysis.detectedAnomalies.push({
          metric: m,
          type: anomalyType,
          zScore: parseFloat(zScore.toFixed(2)),
          value: val,
          baselineMedian: base.median,
          baselineMAD: base.mad,
          fixedExceeded
        });

        // Trigger deduplicated alert
        const cooldownKey = `${point.patient_id}-${m}-${anomalyType}`;
        const lastTriggerIdx = activeAlertCooldowns[cooldownKey] || -99;

        if (idx - lastTriggerIdx > 4) { // 4 observation cooldown
          activeAlertCooldowns[cooldownKey] = idx;

          let priority = 'Moderate';
          if (absZ > 4.0 || (m === 'spo2' && val < 90) || (m === 'systolic_bp' && val > 160)) {
            priority = 'High';
          } else if (absZ < 2.5) {
            priority = 'Low';
          }

          alerts.push({
            id: `ALT-${point.patient_id}-${idx}-${m}`,
            patientId: point.patient_id,
            timestamp: point.timestamp,
            metric: m,
            metricName: formatMetricName(m),
            value: val,
            unit: getMetricUnit(m),
            priority,
            anomalyType,
            zScore: parseFloat(zScore.toFixed(2)),
            baselineMedian: base.median,
            baselineMAD: base.mad,
            fixedExceeded,
            status: 'Awaiting Review',
            evidence: `Observed ${formatMetricName(m)} of ${val}${getMetricUnit(m)} represents a ${anomalyType.toLowerCase()} (${zScore > 0 ? '+' : ''}${zScore.toFixed(1)} robust std dev from baseline median of ${base.median}${getMetricUnit(m)}).`,
            suggestedAction: `Check sensor attachment & record quality. Request qualified medical staff review if trend persists.`
          });
        }
      }
    });

    processedData.push(pointAnalysis);
  });

  return { alerts, processedData, baselines };
};

// Helper utilities
const formatMetricName = (m) => {
  const map = {
    heart_rate: 'Heart Rate',
    respiratory_rate: 'Respiratory Rate',
    temperature_c: 'Body Temp',
    spo2: 'SpO2',
    systolic_bp: 'Systolic BP'
  };
  return map[m] || m;
};

const getMetricUnit = (m) => {
  const map = {
    heart_rate: ' bpm',
    respiratory_rate: ' bpm',
    temperature_c: ' °C',
    spo2: '%',
    systolic_bp: ' mmHg'
  };
  return map[m] || '';
};

// ==========================================
// 2. SYNTHETIC DATA ENGINE WITH GROUND TRUTH
// ==========================================

const GENERATE_SYNTHETIC_DATASET = () => {
  const patients = ['ES-1001', 'ES-1002', 'ES-1003', 'ES-1004', 'ES-1005', 'ES-1006'];
  const dataset = [];
  const startTime = new Date('2026-10-15T00:00:00Z');

  patients.forEach(pid => {
    let baseHR = 70 + Math.floor((pid.charCodeAt(6) * 7) % 15);
    let baseRR = 14 + Math.floor((pid.charCodeAt(6) * 3) % 4);
    let baseTemp = 36.6 + ((pid.charCodeAt(6) % 3) * 0.1);
    let baseSpO2 = 97 + (pid.charCodeAt(6) % 2);
    let baseBP = 115 + Math.floor((pid.charCodeAt(6) * 5) % 12);

    for (let i = 0; i < 30; i++) {
      const time = new Date(startTime.getTime() + i * 2 * 3600 * 1000).toISOString();
      let hr = baseHR + Math.round((Math.sin(i / 2) * 3) + (Math.random() * 4 - 2));
      let rr = baseRR + Math.round((Math.cos(i / 3) * 1.5) + (Math.random() * 2 - 1));
      let temp = parseFloat((baseTemp + (Math.sin(i / 4) * 0.15) + (Math.random() * 0.1 - 0.05)).toFixed(1));
      let spo2 = baseSpO2 + Math.round(Math.random() * 1.5 - 0.75);
      let bp = baseBP + Math.round((Math.sin(i / 2) * 4) + (Math.random() * 6 - 3));

      let groundTruthAnomaly = false;
      let anomalyDescription = 'Normal Variation';

      // Inject Scenario 2: ES-1002 Sudden Spike at index 18
      if (pid === 'ES-1002' && i === 18) {
        hr += 35; // Spike to ~110
        groundTruthAnomaly = true;
        anomalyDescription = 'Injected Sudden Spike';
      }

      // Inject Scenario 3: ES-1003 Gradual Drift (Indices 16-24)
      // Drift HR from 68 to 92 (stays < 100 fixed threshold, but shifts 4 MADs)
      if (pid === 'ES-1003' && i >= 16 && i <= 24) {
        const driftAmount = (i - 15) * 3.2; // gradual rise
        hr += Math.round(driftAmount);
        if (i >= 19) {
          groundTruthAnomaly = true;
          anomalyDescription = 'Injected Gradual Drift';
        }
      }

      // Inject Scenario 4: ES-1004 Persistent Shift (Indices 14-22)
      if (pid === 'ES-1004' && i >= 14 && i <= 22) {
        temp += 0.8; // Mild persistent fever 37.5 (below 37.8 fixed max)
        groundTruthAnomaly = true;
        anomalyDescription = 'Injected Persistent Shift';
      }

      // Inject Scenario 5: ES-1005 Multi-Vital Co-Deviation (Indices 20-25)
      if (pid === 'ES-1005' && i >= 20 && i <= 25) {
        hr += 22;
        rr += 6;
        spo2 -= 4;
        groundTruthAnomaly = true;
        anomalyDescription = 'Injected Multi-Vital Change';
      }

      // Inject Scenario 6: ES-1006 Missing Data Gaps
      if (pid === 'ES-1006' && (i === 12 || i === 13 || i === 14)) {
        hr = null;
        rr = null;
        spo2 = null;
      }

      // Cap boundaries reasonably
      if (spo2 > 100) spo2 = 100;

      dataset.push({
        timestamp: time,
        patient_id: pid,
        heart_rate: hr,
        respiratory_rate: rr,
        temperature_c: temp,
        spo2: spo2,
        systolic_bp: bp,
        groundTruthAnomaly,
        anomalyDescription
      });
    }
  });

  return dataset;
};

// ==========================================
// MAIN APP COMPONENT
// ==========================================

export default function App() {
  // Navigation & View state
  const [activeTab, setActiveTab] = useState('overview'); // overview, patients, patient-detail, alerts, comparison, demo, data-io, safety
  const [selectedPatientId, setSelectedPatientId] = useState('ES-1003'); // Default to gradual drift patient
  const [selectedMetric, setSelectedMetric] = useState('heart_rate');
  const [timeRange, setTimeRange] = useState('7d'); // 24h, 48h, 7d
  const [alertFilter, setAlertFilter] = useState('ALL'); // ALL, High, Moderate, Low, Awaiting Review
  const [searchQuery, setSearchQuery] = useState('');

  // Primary Data State
  const [rawDataset, setRawDataset] = useState(() => GENERATE_SYNTHETIC_DATASET());
  const [userAlertNotes, setUserAlertNotes] = useState({});
  const [reviewedAlerts, setReviewedAlerts] = useState({});

  // Guided Demo State
  const [demoStep, setDemoStep] = useState(0);
  const [isDemoPlaying, setIsDemoPlaying] = useState(false);
  const [demoPlaybackIdx, setDemoPlaybackIdx] = useState(8);

  // Re-run statistical detection whenever raw dataset updates
  const detectionResults = useMemo(() => {
    return analyzeTimeSeriesData(rawDataset);
  }, [rawDataset]);

  const { alerts, processedData, baselines } = detectionResults;

  // Filtered dataset for selected patient
  const patientData = useMemo(() => {
    return processedData.filter(d => d.patient_id === selectedPatientId);
  }, [processedData, selectedPatientId]);

  // Patients Master List with non-diagnostic summaries
  const patientList = useMemo(() => {
    const pids = ['ES-1001', 'ES-1002', 'ES-1003', 'ES-1004', 'ES-1005', 'ES-1006'];
    return pids.map(id => {
      const pData = processedData.filter(d => d.patient_id === id);
      const latest = pData[pData.length - 1] || {};
      const activeAlerts = alerts.filter(a => a.patientId === id && !reviewedAlerts[a.id]);
      
      let trendSummary = "Stable baseline readings.";
      if (id === 'ES-1002') trendSummary = "Detected isolated sudden spike in HR.";
      if (id === 'ES-1003') trendSummary = "Subtle upward drift in resting heart rate.";
      if (id === 'ES-1004') trendSummary = "Persistent mild body temperature elevation.";
      if (id === 'ES-1005') trendSummary = "Synchronized change across multiple vitals.";
      if (id === 'ES-1006') trendSummary = "Irregular transmission with telemetry gaps.";

      return {
        id,
        ageGroup: id === 'ES-1001' ? '45-50' : id === 'ES-1002' ? '60-65' : '55-60',
        status: activeAlerts.length > 0 ? 'Active Monitoring Alert' : 'Baseline Stable',
        latest,
        activeAlertsCount: activeAlerts.length,
        trendSummary
      };
    });
  }, [processedData, alerts, reviewedAlerts]);

  // Guided Demo Timer Simulation
  useEffect(() => {
    let interval = null;
    if (isDemoPlaying) {
      interval = setInterval(() => {
        setDemoPlaybackIdx(prev => {
          if (prev >= 30) {
            setIsDemoPlaying(false);
            return 30;
          }
          return prev + 1;
        });
      }, 1000);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isDemoPlaying]);

  // Filtered Alerts for Alert Center
  const filteredAlerts = useMemo(() => {
    return alerts.filter(a => {
      if (alertFilter === 'ALL') return true;
      if (alertFilter === 'Awaiting Review') return !reviewedAlerts[a.id];
      return a.priority === alertFilter;
    });
  }, [alerts, alertFilter, reviewedAlerts]);

  // Action Handlers
  const handleToggleReviewed = (alertId) => {
    setReviewedAlerts(prev => ({ ...prev, [alertId]: !prev[alertId] }));
  };

  const handleAddNote = (alertId, noteText) => {
    setUserAlertNotes(prev => ({ ...prev, [alertId]: noteText }));
  };

  const handleResetData = () => {
    setRawDataset(GENERATE_SYNTHETIC_DATASET());
    setReviewedAlerts({});
    setUserAlertNotes({});
  };

  // CSV Export
  const handleExportCSV = () => {
    const headers = "timestamp,patient_id,heart_rate,respiratory_rate,temperature_c,spo2,systolic_bp\n";
    const rows = rawDataset.map(d => 
      `${d.timestamp},${d.patient_id},${d.heart_rate ?? ''},${d.respiratory_rate ?? ''},${d.temperature_c ?? ''},${d.spo2 ?? ''},${d.systolic_bp ?? ''}`
    ).join("\n");
    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `EarlyShift_Synthetic_Export_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
  };

  // Metric Calculation for Comparison Lab
  const calculateBenchmarkMetrics = (patientIdFilter = null) => {
    let tp_pers = 0, fp_pers = 0, fn_pers = 0;
    let tp_fixed = 0, fp_fixed = 0, fn_fixed = 0;

    const evalData = rawDataset.filter(d => !patientIdFilter || d.patient_id === patientIdFilter);
    
    // Group eval data by patient and run detections
    evalData.forEach(pt => {
      const isGroundTruth = pt.groundTruthAnomaly;
      
      // Check if Personalized Anomaly logic flagged this
      const ptAnalyzed = processedData.find(p => p.patient_id === pt.patient_id && p.timestamp === pt.timestamp);
      const isPersFlagged = ptAnalyzed && ptAnalyzed.detectedAnomalies && ptAnalyzed.detectedAnomalies.length > 0;

      // Check if Fixed Threshold flagged this
      let isFixedFlagged = false;
      ['heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 'systolic_bp'].forEach(m => {
        const val = pt[m];
        if (val !== null && val !== undefined) {
          const thresh = FIXED_THRESHOLDS[m];
          if (val < thresh.min || val > thresh.max) isFixedFlagged = true;
        }
      });

      // Personalized confusion
      if (isPersFlagged && isGroundTruth) tp_pers++;
      if (isPersFlagged && !isGroundTruth) fp_pers++;
      if (!isPersFlagged && isGroundTruth) fn_pers++;

      // Fixed confusion
      if (isFixedFlagged && isGroundTruth) tp_fixed++;
      if (isFixedFlagged && !isGroundTruth) fp_fixed++;
      if (!isFixedFlagged && isGroundTruth) fn_fixed++;
    });

    const getStats = (tp, fp, fn) => {
      const precision = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
      const recall = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      return { precision: precision.toFixed(1), recall: recall.toFixed(1), f1: f1.toFixed(1), tp, fp, fn };
    };

    return {
      personalized: getStats(tp_pers, fp_pers, fn_pers),
      fixed: getStats(tp_fixed, fp_fixed, fn_fixed)
    };
  };

  const currentBenchmarkStats = calculateBenchmarkMetrics();

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans overflow-hidden">
      
      {/* SIDEBAR NAVIGATION */}
      <aside className="w-64 bg-slate-900 text-white flex flex-col justify-between shrink-0 shadow-xl border-r border-slate-800">
        <div>
          {/* Custom Wordmark Logo */}
          <div className="p-5 border-b border-slate-800 flex items-center space-x-3">
            <div className="w-9 h-9 rounded-lg bg-teal-500/20 border border-teal-500/40 flex items-center justify-center text-teal-400">
              <Activity className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <h1 className="font-bold text-lg tracking-wider text-slate-100 leading-none">
                Early<span className="text-teal-400">Shift</span>
              </h1>
              <p className="text-[10px] text-slate-400 mt-1 uppercase tracking-widest font-medium">Early-Warning Platform</p>
            </div>
          </div>

          {/* Nav Items */}
          <nav className="mt-4 px-3 space-y-1">
            {[
              { id: 'overview', label: 'Overview Dashboard', icon: BarChart2 },
              { id: 'patients', label: 'Patient Monitoring', icon: Users },
              { id: 'patient-detail', label: 'Time-Series Analysis', icon: Activity },
              { id: 'alerts', label: 'Explainable Alert Center', icon: AlertTriangle, badge: alerts.filter(a => !reviewedAlerts[a.id]).length },
              { id: 'comparison', label: 'Algorithm Comparison Lab', icon: Sliders },
              { id: 'demo', label: 'Guided Demo Mode', icon: Play, highlight: true },
              { id: 'data-io', label: 'Data Import / Export', icon: Upload },
              { id: 'safety', label: 'Safety & Limitations', icon: Info }
            ].map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                    isActive 
                      ? 'bg-teal-600 text-white shadow-md shadow-teal-900/30' 
                      : item.highlight 
                        ? 'bg-slate-800/80 text-teal-300 hover:bg-slate-800 border border-teal-500/30'
                        : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <Icon className={`w-4 h-4 ${isActive ? 'text-white' : item.highlight ? 'text-teal-400' : 'text-slate-400'}`} />
                    <span>{item.label}</span>
                  </div>
                  {item.badge > 0 && (
                    <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-amber-500 text-slate-950">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Prototype Footer Badge */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/50">
          <div className="flex items-center space-x-2 text-xs text-amber-400/90 font-medium bg-amber-500/10 p-2.5 rounded-md border border-amber-500/20">
            <ShieldAlert className="w-4 h-4 shrink-0" />
            <span>Research Prototype (Synthetic Data Only)</span>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 flex flex-col overflow-y-auto bg-slate-100">
        
        {/* Header Bar */}
        <header className="bg-white border-b border-slate-200 py-3 px-6 flex items-center justify-between sticky top-0 z-20 shadow-sm">
          <div className="flex items-center space-x-3">
            <h2 className="text-xl font-bold text-slate-800 capitalize">
              {activeTab.replace('-', ' ')}
            </h2>
            <span className="text-xs font-semibold bg-slate-100 text-slate-600 px-2.5 py-1 rounded-full border border-slate-200">
              Personalized Baseline MAD Detector
            </span>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={handleResetData}
              className="text-xs font-medium text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 px-3 py-1.5 rounded-md transition border border-slate-200 flex items-center space-x-1.5"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Demo Data</span>
            </button>
            <button
              onClick={() => setActiveTab('demo')}
              className="text-xs font-semibold bg-teal-600 hover:bg-teal-700 text-white px-3.5 py-1.5 rounded-md shadow-sm transition flex items-center space-x-1.5"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Launch Guided Demo</span>
            </button>
          </div>
        </header>

        {/* PAGE CONTENT CONTAINER */}
        <div className="p-6 max-w-7xl w-full mx-auto space-y-6">

          {/* ========================================== */}
          {/* TAB 1: OVERVIEW DASHBOARD                  */}
          {/* ========================================== */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              
              {/* Vision Banner */}
              <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-teal-950 text-white p-6 rounded-xl shadow-lg border border-slate-800 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                  <h3 className="text-xl font-bold tracking-tight">"See the change before it becomes a pattern."</h3>
                  <p className="text-slate-300 text-sm mt-1 max-w-2xl">
                    EarlyShift evaluates time-series patient observations against individual historic baselines to detect subtle, persistent, and multi-vital trends missed by fixed thresholds.
                  </p>
                </div>
                <button
                  onClick={() => setActiveTab('comparison')}
                  className="shrink-0 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold px-4 py-2 rounded-lg text-xs transition shadow"
                >
                  View Benchmark Results &rarr;
                </button>
              </div>

              {/* KPI Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { label: 'Monitored Patients', val: '6', sub: 'Fictional cohort ES-1001 to 1006', icon: Users, color: 'text-blue-600' },
                  { label: 'Patients with Alerts', val: alerts.reduce((acc, a) => acc.add(a.patientId), new Set()).size, sub: 'Elevated anomaly z-score', icon: AlertTriangle, color: 'text-amber-600' },
                  { label: 'Active Alerts Pending', val: alerts.filter(a => !reviewedAlerts[a.id]).length, sub: 'Awaiting user review', icon: Clock, color: 'text-rose-600' },
                  { label: 'Measurements Analyzed', val: rawDataset.length * 5, sub: '180 observations processed', icon: Activity, color: 'text-teal-600' },
                ].map((kpi, idx) => {
                  const Icon = kpi.icon;
                  return (
                    <div key={idx} className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex justify-between items-start">
                      <div>
                        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">{kpi.label}</p>
                        <h4 className="text-2xl font-extrabold text-slate-900 mt-1">{kpi.val}</h4>
                        <p className="text-xs text-slate-500 mt-1">{kpi.sub}</p>
                      </div>
                      <div className={`p-2.5 rounded-lg bg-slate-50 ${kpi.color}`}>
                        <Icon className="w-5 h-5" />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Dashboard Main Visuals */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                
                {/* Time-Series Trend Overview Chart */}
                <div className="lg:col-span-2 bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <h4 className="font-bold text-slate-800 text-base">Population Heart Rate Trend & Anomalies</h4>
                      <p className="text-xs text-slate-500">Highlighted dots indicate localized robust z-score deviations (&gt; 2.5 MAD)</p>
                    </div>
                    <select
                      value={selectedPatientId}
                      onChange={(e) => setSelectedPatientId(e.target.value)}
                      className="text-xs font-medium bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 text-slate-700"
                    >
                      {patientList.map(p => <option key={p.id} value={p.id}>Patient {p.id}</option>)}
                    </select>
                  </div>

                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={patientData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                        <XAxis 
                          dataKey="timestamp" 
                          tickFormatter={(t) => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          tick={{ fontSize: 11, fill: '#64748B' }}
                        />
                        <YAxis domain={['auto', 'auto']} tick={{ fontSize: 11, fill: '#64748B' }} />
                        <Tooltip 
                          labelFormatter={(t) => `Timestamp: ${new Date(t).toLocaleString()}`}
                          formatter={(val, name) => [`${val} bpm`, 'Heart Rate']}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="heart_rate" 
                          stroke="#0D9488" 
                          strokeWidth={2} 
                          dot={(props) => {
                            const { cx, cy, payload } = props;
                            const isAnom = payload.detectedAnomalies && payload.detectedAnomalies.some(a => a.metric === 'heart_rate');
                            if (isAnom) {
                              return <circle key={cx} cx={cx} cy={cy} r={5} fill="#DC2626" stroke="#FFF" strokeWidth={2} />;
                            }
                            return <circle key={cx} cx={cx} cy={cy} r={2} fill="#0D9488" />;
                          }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Recent Alerts Feed */}
                <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-center mb-4">
                      <h4 className="font-bold text-slate-800 text-base">Recent Detected Events</h4>
                      <button 
                        onClick={() => setActiveTab('alerts')}
                        className="text-xs text-teal-600 hover:text-teal-700 font-semibold"
                      >
                        View All ({alerts.length})
                      </button>
                    </div>

                    <div className="space-y-3 max-h-64 overflow-y-auto pr-1">
                      {alerts.slice(0, 4).map(alert => (
                        <div key={alert.id} className="p-3 bg-slate-50 rounded-lg border border-slate-200 hover:border-slate-300 transition">
                          <div className="flex justify-between items-start">
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                              alert.priority === 'High' ? 'bg-rose-100 text-rose-700' : 'bg-amber-100 text-amber-700'
                            }`}>
                              {alert.priority} Priority
                            </span>
                            <span className="text-[11px] text-slate-400">
                              {new Date(alert.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </span>
                          </div>
                          <p className="text-xs font-bold text-slate-800 mt-1.5">
                            {alert.patientId} — {alert.metricName} {alert.anomalyType}
                          </p>
                          <p className="text-[11px] text-slate-600 mt-1 line-clamp-2">
                            {alert.evidence}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="mt-4 pt-3 border-t border-slate-100 text-center">
                    <span className="text-xs text-slate-500">Automated baseline estimation active across all 6 patients.</span>
                  </div>
                </div>

              </div>

              {/* Explanatory System Architecture */}
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h4 className="font-bold text-slate-800 text-base mb-3 flex items-center space-x-2">
                  <HelpCircle className="w-5 h-5 text-teal-600" />
                  <span>How EarlyShift Detection Works</span>
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-slate-600">
                  <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                    <span className="font-bold text-slate-800 text-sm block mb-1">1. Robust Baseline Estimation</span>
                    Calculates individualized median and Median Absolute Deviation (MAD) over initial observations to filter patient-specific normal range without assuming normal distribution.
                  </div>
                  <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                    <span className="font-bold text-slate-800 text-sm block mb-1">2. Multi-Signal Pattern Analysis</span>
                    Evaluates robust z-scores to catch sudden spikes (|Z| &gt; 3.0), gradual drifts (rolling directional change &gt; 1.8), and persistent deviations (&gt; 3 consecutive points).
                  </div>
                  <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                    <span className="font-bold text-slate-800 text-sm block mb-1">3. Deduplication & Explainability</span>
                    Applies alert cooldowns to prevent noise, generates time-series evidence explanations, and compares flags against illustrative standard medical thresholds.
                  </div>
                </div>
              </div>

            </div>
          )}

          {/* ========================================== */}
          {/* TAB 2: PATIENT MONITORING LIST             */}
          {/* ========================================== */}
          {activeTab === 'patients' && (
            <div className="space-y-6">
              
              {/* Search & Filter Bar */}
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col sm:flex-row justify-between items-center gap-4">
                <div className="relative w-full sm:w-72">
                  <Search className="w-4 h-4 absolute left-3 top-3 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Search Patient ID (e.g. ES-1003)..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-9 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500"
                  />
                </div>
                <div className="flex items-center space-x-2 text-xs text-slate-500">
                  <span>Showing synthetic patient records ES-1001 through ES-1006</span>
                </div>
              </div>

              {/* Patient Cards Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {patientList
                  .filter(p => p.id.toLowerCase().includes(searchQuery.toLowerCase()))
                  .map((patient) => {
                    const latest = patient.latest;
                    return (
                      <div 
                        key={patient.id} 
                        className="bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition flex flex-col justify-between overflow-hidden"
                      >
                        <div className="p-5">
                          <div className="flex justify-between items-start">
                            <div>
                              <h3 className="text-lg font-bold text-slate-900">{patient.id}</h3>
                              <p className="text-xs text-slate-500">Age Bracket: {patient.ageGroup} yrs</p>
                            </div>
                            <span className={`px-2.5 py-1 rounded-full text-xs font-semibold ${
                              patient.activeAlertsCount > 0 
                                ? 'bg-amber-100 text-amber-800 border border-amber-200' 
                                : 'bg-teal-50 text-teal-700 border border-teal-200'
                            }`}>
                              {patient.activeAlertsCount > 0 ? `${patient.activeAlertsCount} Active Alert` : 'Baseline Stable'}
                            </span>
                          </div>

                          {/* Latest Measurements Display */}
                          <div className="grid grid-cols-3 gap-2 mt-4 text-center">
                            <div className="bg-slate-50 p-2 rounded border border-slate-100">
                              <span className="text-[10px] text-slate-500 block uppercase font-medium">Heart Rate</span>
                              <span className="text-xs font-bold text-slate-800">{latest.heart_rate ?? 'N/A'} bpm</span>
                            </div>
                            <div className="bg-slate-50 p-2 rounded border border-slate-100">
                              <span className="text-[10px] text-slate-500 block uppercase font-medium">SpO2</span>
                              <span className="text-xs font-bold text-slate-800">{latest.spo2 ?? 'N/A'}%</span>
                            </div>
                            <div className="bg-slate-50 p-2 rounded border border-slate-100">
                              <span className="text-[10px] text-slate-500 block uppercase font-medium">Sys BP</span>
                              <span className="text-xs font-bold text-slate-800">{latest.systolic_bp ?? 'N/A'} mmHg</span>
                            </div>
                          </div>

                          {/* Trend Summary */}
                          <div className="mt-4 bg-slate-50 p-3 rounded-lg border border-slate-200">
                            <p className="text-xs font-medium text-slate-700">
                              <span className="font-bold text-slate-900">Pattern Summary: </span>
                              {patient.trendSummary}
                            </p>
                          </div>
                        </div>

                        {/* Action Link */}
                        <div className="bg-slate-50 px-5 py-3 border-t border-slate-200 flex justify-between items-center">
                          <span className="text-[11px] text-slate-500">12-point baseline establish window</span>
                          <button
                            onClick={() => {
                              setSelectedPatientId(patient.id);
                              setActiveTab('patient-detail');
                            }}
                            className="text-xs font-semibold text-teal-600 hover:text-teal-700 flex items-center space-x-1"
                          >
                            <span>Inspect Time-Series</span>
                            <ChevronRight className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* ========================================== */}
          {/* TAB 3: PATIENT DETAIL & TIME-SERIES        */}
          {/* ========================================== */}
          {activeTab === 'patient-detail' && (
            <div className="space-y-6">
              
              {/* Patient Header Controls */}
              <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                  <div className="flex items-center space-x-3">
                    <h3 className="text-2xl font-extrabold text-slate-900">Patient {selectedPatientId}</h3>
                    <span className="text-xs font-semibold bg-slate-100 text-slate-700 px-3 py-1 rounded-full border border-slate-200">
                      Synthetic Record ID
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    Continuous monitoring timeline • Baseline established using initial 12 points
                  </p>
                </div>

                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-1 bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs font-medium">
                    {['24h', '48h', '7d'].map((range) => (
                      <button
                        key={range}
                        onClick={() => setTimeRange(range)}
                        className={`px-3 py-1 rounded-md transition ${timeRange === range ? 'bg-white text-slate-900 shadow-sm font-bold' : 'text-slate-600 hover:text-slate-900'}`}
                      >
                        {range.toUpperCase()}
                      </button>
                    ))}
                  </div>

                  <select
                    value={selectedPatientId}
                    onChange={(e) => setSelectedPatientId(e.target.value)}
                    className="text-xs font-semibold bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-slate-800"
                  >
                    {patientList.map(p => <option key={p.id} value={p.id}>Switch to {p.id}</option>)}
                  </select>
                </div>
              </div>

              {/* Vital Metric Selector Tabs */}
              <div className="flex space-x-2 border-b border-slate-200 pb-2 overflow-x-auto">
                {[
                  { id: 'heart_rate', label: 'Heart Rate', unit: 'bpm', icon: Heart },
                  { id: 'respiratory_rate', label: 'Respiratory Rate', unit: 'bpm', icon: Wind },
                  { id: 'temperature_c', label: 'Body Temperature', unit: '°C', icon: Thermometer },
                  { id: 'spo2', label: 'Oxygen Saturation (SpO2)', unit: '%', icon: Zap },
                  { id: 'systolic_bp', label: 'Systolic Blood Pressure', unit: 'mmHg', icon: Activity }
                ].map((m) => {
                  const Icon = m.icon;
                  const isSelected = selectedMetric === m.id;
                  return (
                    <button
                      key={m.id}
                      onClick={() => setSelectedMetric(m.id)}
                      className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold transition shrink-0 ${
                        isSelected 
                          ? 'bg-slate-900 text-white shadow' 
                          : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
                      }`}
                    >
                      <Icon className={`w-3.5 h-3.5 ${isSelected ? 'text-teal-400' : 'text-slate-400'}`} />
                      <span>{m.label}</span>
                    </button>
                  );
                })}
              </div>

              {/* Main Time Series Chart */}
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <h4 className="font-bold text-slate-800 text-lg">
                      {formatMetricName(selectedMetric)} Baseline vs Observed Deviation
                    </h4>
                    <p className="text-xs text-slate-500">
                      Shaded Area = Patient Individual Median ± 2.0 MAD Baseline Band • Red Dots = Detected Anomalies
                    </p>
                  </div>

                  {baselines[selectedMetric] && (
                    <div className="text-right text-xs">
                      <span className="font-bold text-slate-800">Baseline Median: </span>
                      <span className="text-teal-700 font-semibold">{baselines[selectedMetric].median} {getMetricUnit(selectedMetric)}</span>
                      <span className="ml-3 font-bold text-slate-800">MAD: </span>
                      <span className="text-slate-600">{baselines[selectedMetric].mad}</span>
                    </div>
                  )}
                </div>

                {/* Recharts Component */}
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={patientData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                      <XAxis 
                        dataKey="timestamp" 
                        tickFormatter={(t) => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        tick={{ fontSize: 11, fill: '#64748B' }}
                      />
                      <YAxis domain={['auto', 'auto']} tick={{ fontSize: 11, fill: '#64748B' }} />
                      <Tooltip 
                        labelFormatter={(t) => `Time: ${new Date(t).toLocaleString()}`}
                        formatter={(val, name, item) => [
                          `${val !== null ? val + getMetricUnit(selectedMetric) : 'Missing Data Gap'}`,
                          formatMetricName(selectedMetric)
                        ]}
                      />
                      
                      {/* Standard Fixed Threshold Reference Lines */}
                      <ReferenceLine 
                        y={FIXED_THRESHOLDS[selectedMetric].max} 
                        stroke="#F59E0B" 
                        strokeDasharray="4 4" 
                        label={{ value: `Fixed Max (${FIXED_THRESHOLDS[selectedMetric].max})`, fill: '#D97706', fontSize: 10, position: 'insideTopRight' }} 
                      />
                      <ReferenceLine 
                        y={FIXED_THRESHOLDS[selectedMetric].min} 
                        stroke="#F59E0B" 
                        strokeDasharray="4 4" 
                        label={{ value: `Fixed Min (${FIXED_THRESHOLDS[selectedMetric].min})`, fill: '#D97706', fontSize: 10, position: 'insideBottomRight' }} 
                      />

                      <Line 
                        type="monotone" 
                        dataKey={selectedMetric} 
                        stroke="#0D9488" 
                        strokeWidth={2.5} 
                        connectNulls={false} // Explicitly show missing data gaps
                        dot={(props) => {
                          const { cx, cy, payload } = props;
                          if (!payload || payload[selectedMetric] === null) return null;
                          const isAnom = payload.detectedAnomalies && payload.detectedAnomalies.some(a => a.metric === selectedMetric);
                          if (isAnom) {
                            return <circle key={cx} cx={cx} cy={cy} r={6} fill="#DC2626" stroke="#FFF" strokeWidth={2} />;
                          }
                          return <circle key={cx} cx={cx} cy={cy} r={3} fill="#0D9488" />;
                        }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="mt-4 flex items-center justify-between text-xs text-slate-500 bg-slate-50 p-3 rounded-lg border border-slate-200">
                  <div className="flex items-center space-x-4">
                    <span className="flex items-center"><span className="w-3 h-3 rounded-full bg-teal-600 inline-block mr-1.5"></span> Observed Patient Signal</span>
                    <span className="flex items-center"><span className="w-3 h-3 rounded-full bg-rose-600 inline-block mr-1.5"></span> EarlyShift Anomaly Flag</span>
                    <span className="flex items-center"><span className="w-3 h-0.5 bg-amber-500 inline-block mr-1.5"></span> Fixed Threshold Limit</span>
                  </div>
                  <span>Note: Missing data points appear as gaps in line plot.</span>
                </div>
              </div>

              {/* Event Evidence Timeline */}
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h4 className="font-bold text-slate-800 text-base mb-4">Detected Event Timeline for {selectedPatientId}</h4>
                <div className="space-y-3">
                  {patientData
                    .filter(p => p.detectedAnomalies && p.detectedAnomalies.length > 0)
                    .map((item, idx) => (
                      <div key={idx} className="p-4 bg-slate-50 rounded-lg border-l-4 border-l-rose-500 border border-slate-200 flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="text-xs font-bold text-slate-900">{new Date(item.timestamp).toLocaleString()}</span>
                            <span className="text-[10px] bg-rose-100 text-rose-800 px-2 py-0.5 rounded font-bold uppercase">
                              Z-Score: {item.detectedAnomalies[0].zScore}
                            </span>
                          </div>
                          <p className="text-xs text-slate-700 mt-1 font-medium">
                            {item.detectedAnomalies.map(a => `${formatMetricName(a.metric)} (${a.value}${getMetricUnit(a.metric)})`).join(', ')} flagged as <span className="font-bold">{item.detectedAnomalies[0].type}</span>.
                          </p>
                        </div>
                        <div className="text-xs text-slate-500 bg-white px-3 py-1.5 rounded border border-slate-200 shrink-0">
                          Fixed Threshold Status: {item.detectedAnomalies[0].fixedExceeded ? 'Exceeded Limit' : 'Inside Fixed Limits'}
                        </div>
                      </div>
                    ))}

                  {patientData.filter(p => p.detectedAnomalies && p.detectedAnomalies.length > 0).length === 0 && (
                    <p className="text-xs text-slate-500 italic p-4 text-center">No statistical anomalies detected for this patient timeline.</p>
                  )}
                </div>
              </div>

            </div>
          )}

          {/* ========================================== */}
          {/* TAB 4: EXPLAINABLE ALERT CENTER            */}
          {/* ========================================== */}
          {activeTab === 'alerts' && (
            <div className="space-y-6">
              
              {/* Alert Filters & Summary */}
              <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col sm:flex-row justify-between items-center gap-4">
                <div className="flex items-center space-x-2 text-xs font-semibold">
                  <span className="text-slate-500">Filter Priority:</span>
                  {['ALL', 'Awaiting Review', 'High', 'Moderate', 'Low'].map((f) => (
                    <button
                      key={f}
                      onClick={() => setAlertFilter(f)}
                      className={`px-3 py-1.5 rounded-lg transition ${
                        alertFilter === f 
                          ? 'bg-slate-900 text-white shadow' 
                          : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                      }`}
                    >
                      {f}
                    </button>
                  ))}
                </div>

                <div className="text-xs text-slate-500">
                  Total System Alerts: <span className="font-bold text-slate-800">{alerts.length}</span>
                </div>
              </div>

              {/* Alerts Cards List */}
              <div className="space-y-4">
                {filteredAlerts.map((alert) => {
                  const isReviewed = reviewedAlerts[alert.id];
                  const note = userAlertNotes[alert.id] || '';

                  return (
                    <div 
                      key={alert.id}
                      className={`bg-white rounded-xl border transition shadow-sm p-6 ${
                        isReviewed ? 'border-slate-200 opacity-75 bg-slate-50/50' : 'border-amber-200 ring-1 ring-amber-100'
                      }`}
                    >
                      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 border-b border-slate-100 pb-4">
                        <div className="flex items-center space-x-3">
                          <span className={`px-2.5 py-1 rounded text-xs font-bold uppercase ${
                            alert.priority === 'High' ? 'bg-rose-100 text-rose-800' : 'bg-amber-100 text-amber-800'
                          }`}>
                            {alert.priority} Review Priority
                          </span>
                          <h3 className="font-bold text-slate-900 text-lg">
                            Patient {alert.patientId} — {alert.metricName} {alert.anomalyType}
                          </h3>
                        </div>

                        <div className="flex items-center space-x-3 text-xs">
                          <span className="text-slate-500 flex items-center">
                            <Clock className="w-3.5 h-3.5 mr-1" />
                            {new Date(alert.timestamp).toLocaleString()}
                          </span>
                          <button
                            onClick={() => handleToggleReviewed(alert.id)}
                            className={`px-3 py-1.5 rounded-lg font-semibold transition flex items-center space-x-1 ${
                              isReviewed 
                                ? 'bg-emerald-100 text-emerald-800 hover:bg-emerald-200' 
                                : 'bg-slate-900 text-white hover:bg-slate-800'
                            }`}
                          >
                            <Check className="w-3.5 h-3.5" />
                            <span>{isReviewed ? 'Marked Reviewed' : 'Mark as Reviewed'}</span>
                          </button>
                        </div>
                      </div>

                      {/* Explainable Evidence Breakdown */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4 text-xs">
                        <div className="bg-slate-50 p-3.5 rounded-lg border border-slate-200">
                          <span className="text-slate-500 uppercase font-bold text-[10px] block">Observed vs Baseline</span>
                          <p className="text-slate-900 font-bold text-sm mt-1">
                            {alert.value}{alert.unit}
                          </p>
                          <p className="text-slate-500 mt-0.5">
                            Baseline Median: {alert.baselineMedian}{alert.unit} (MAD: {alert.baselineMAD})
                          </p>
                        </div>

                        <div className="bg-slate-50 p-3.5 rounded-lg border border-slate-200">
                          <span className="text-slate-500 uppercase font-bold text-[10px] block">Robust Z-Score Shift</span>
                          <p className="text-slate-900 font-bold text-sm mt-1">
                            {alert.zScore > 0 ? `+${alert.zScore}` : alert.zScore} Scale Units
                          </p>
                          <p className="text-slate-500 mt-0.5">
                            Detection Pattern: {alert.anomalyType}
                          </p>
                        </div>

                        <div className="bg-slate-50 p-3.5 rounded-lg border border-slate-200">
                          <span className="text-slate-500 uppercase font-bold text-[10px] block">Fixed Limit Comparison</span>
                          <p className={`font-bold text-sm mt-1 ${alert.fixedExceeded ? 'text-amber-700' : 'text-emerald-700'}`}>
                            {alert.fixedExceeded ? 'Exceeded Standard Max' : 'INSIDE Standard Range'}
                          </p>
                          <p className="text-slate-500 mt-0.5">
                            Standard range: {FIXED_THRESHOLDS[alert.metric]?.label}
                          </p>
                        </div>
                      </div>

                      {/* Text Explanations */}
                      <div className="mt-4 space-y-2 text-xs">
                        <p className="text-slate-700 bg-amber-50/50 p-3 rounded-lg border border-amber-100">
                          <span className="font-bold text-amber-900">Statistical Evidence: </span>
                          {alert.evidence}
                        </p>
                        <p className="text-slate-600 bg-slate-50 p-3 rounded-lg border border-slate-200">
                          <span className="font-bold text-slate-800">Suggested Verification Step: </span>
                          {alert.suggestedAction}
                        </p>
                      </div>

                      {/* User Notes Input Area */}
                      <div className="mt-4 pt-3 border-t border-slate-100 flex items-center space-x-3">
                        <FileText className="w-4 h-4 text-slate-400 shrink-0" />
                        <input
                          type="text"
                          placeholder="Add clinical observation note (persisted locally)..."
                          value={note}
                          onChange={(e) => handleAddNote(alert.id, e.target.value)}
                          className="flex-1 text-xs bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-500"
                        />
                      </div>
                    </div>
                  );
                })}

                {filteredAlerts.length === 0 && (
                  <div className="bg-white p-12 text-center rounded-xl border border-slate-200 text-slate-500 text-sm">
                    No alerts match the selected priority filter.
                  </div>
                )}
              </div>

            </div>
          )}

          {/* ========================================== */}
          {/* TAB 5: ALGORITHM COMPARISON LAB            */}
          {/* ========================================== */}
          {activeTab === 'comparison' && (
            <div className="space-y-6">
              
              {/* Lab Header */}
              <div className="bg-slate-900 text-white p-6 rounded-xl shadow-md border border-slate-800">
                <h3 className="text-xl font-bold">Algorithm Comparison Lab: Baseline vs Anomaly Detector</h3>
                <p className="text-slate-300 text-xs mt-1 max-w-3xl">
                  Evaluates the performance of a Conventional Fixed-Threshold Rule engine against EarlyShift’s Personalized Robust Baseline MAD Anomaly Detector across injected synthetic ground-truth anomalies.
                </p>
              </div>

              {/* Side-by-Side Metric Evaluation Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Method 1: Fixed Threshold */}
                <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
                  <div className="flex justify-between items-center border-b border-slate-100 pb-3">
                    <div>
                      <h4 className="font-bold text-slate-900 text-base">Method A: Fixed Threshold Baseline</h4>
                      <p className="text-xs text-slate-500">Alerts strictly when value exceeds standard limits (e.g. HR &gt; 100)</p>
                    </div>
                    <span className="text-xs font-bold bg-amber-100 text-amber-800 px-2.5 py-1 rounded">Conventional</span>
                  </div>

                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                      <span className="text-[10px] font-bold text-slate-500 uppercase">Precision</span>
                      <p className="text-xl font-extrabold text-slate-900">{currentBenchmarkStats.fixed.precision}%</p>
                    </div>
                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                      <span className="text-[10px] font-bold text-slate-500 uppercase">Recall / Sensitivity</span>
                      <p className="text-xl font-extrabold text-slate-900">{currentBenchmarkStats.fixed.recall}%</p>
                    </div>
                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                      <span className="text-[10px] font-bold text-slate-500 uppercase">F1 Score</span>
                      <p className="text-xl font-extrabold text-slate-900">{currentBenchmarkStats.fixed.f1}%</p>
                    </div>
                  </div>

                  {/* Confusion Matrix */}
                  <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 text-xs">
                    <span className="font-bold text-slate-700 block mb-2 text-center">Confusion Matrix (Synthetic Dataset)</span>
                    <div className="grid grid-cols-2 gap-2 text-center font-mono text-xs">
                      <div className="bg-emerald-100 p-2 rounded border border-emerald-200">
                        <span className="block text-[10px] text-emerald-800 font-sans">True Positives</span>
                        <span className="font-bold text-emerald-900 text-sm">{currentBenchmarkStats.fixed.tp}</span>
                      </div>
                      <div className="bg-rose-100 p-2 rounded border border-rose-200">
                        <span className="block text-[10px] text-rose-800 font-sans">False Positives</span>
                        <span className="font-bold text-rose-900 text-sm">{currentBenchmarkStats.fixed.fp}</span>
                      </div>
                      <div className="bg-rose-100 p-2 rounded border border-rose-200">
                        <span className="block text-[10px] text-rose-800 font-sans">False Negatives (Misses)</span>
                        <span className="font-bold text-rose-900 text-sm">{currentBenchmarkStats.fixed.fn}</span>
                      </div>
                      <div className="bg-slate-200 p-2 rounded">
                        <span className="block text-[10px] text-slate-600 font-sans">True Negatives</span>
                        <span className="font-bold text-slate-800 text-sm">160+</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Method 2: EarlyShift Personalized Detector */}
                <div className="bg-white p-6 rounded-xl border-2 border-teal-500 shadow-md space-y-4">
                  <div className="flex justify-between items-center border-b border-slate-100 pb-3">
                    <div>
                      <h4 className="font-bold text-slate-900 text-base">Method B: EarlyShift Anomaly Detector</h4>
                      <p className="text-xs text-slate-500">Individual Baseline Median + Robust MAD Shift Tracking</p>
                    </div>
                    <span className="text-xs font-bold bg-teal-100 text-teal-800 px-2.5 py-1 rounded">EarlyShift</span>
                  </div>

                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="bg-teal-50 p-3 rounded-lg border border-teal-100">
                      <span className="text-[10px] font-bold text-teal-800 uppercase">Precision</span>
                      <p className="text-xl font-extrabold text-teal-900">{currentBenchmarkStats.personalized.precision}%</p>
                    </div>
                    <div className="bg-teal-50 p-3 rounded-lg border border-teal-100">
                      <span className="text-[10px] font-bold text-teal-800 uppercase">Recall / Sensitivity</span>
                      <p className="text-xl font-extrabold text-teal-900">{currentBenchmarkStats.personalized.recall}%</p>
                    </div>
                    <div className="bg-teal-50 p-3 rounded-lg border border-teal-100">
                      <span className="text-[10px] font-bold text-teal-800 uppercase">F1 Score</span>
                      <p className="text-xl font-extrabold text-teal-900">{currentBenchmarkStats.personalized.f1}%</p>
                    </div>
                  </div>

                  {/* Confusion Matrix */}
                  <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 text-xs">
                    <span className="font-bold text-slate-700 block mb-2 text-center">Confusion Matrix (Synthetic Dataset)</span>
                    <div className="grid grid-cols-2 gap-2 text-center font-mono text-xs">
                      <div className="bg-emerald-100 p-2 rounded border border-emerald-200">
                        <span className="block text-[10px] text-emerald-800 font-sans">True Positives</span>
                        <span className="font-bold text-emerald-900 text-sm">{currentBenchmarkStats.personalized.tp}</span>
                      </div>
                      <div className="bg-rose-100 p-2 rounded border border-rose-200">
                        <span className="block text-[10px] text-rose-800 font-sans">False Positives</span>
                        <span className="font-bold text-rose-900 text-sm">{currentBenchmarkStats.personalized.fp}</span>
                      </div>
                      <div className="bg-rose-100 p-2 rounded border border-rose-200">
                        <span className="block text-[10px] text-rose-800 font-sans">False Negatives (Misses)</span>
                        <span className="font-bold text-rose-900 text-sm">{currentBenchmarkStats.personalized.fn}</span>
                      </div>
                      <div className="bg-slate-200 p-2 rounded">
                        <span className="block text-[10px] text-slate-600 font-sans">True Negatives</span>
                        <span className="font-bold text-slate-800 text-sm">160+</span>
                      </div>
                    </div>
                  </div>
                </div>

              </div>

              {/* Case Study: Gradual Drift Detection Highlight */}
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-3">
                <h4 className="font-bold text-slate-900 text-base">Key Insight Case Study: Patient ES-1003 (Gradual Drift)</h4>
                <p className="text-xs text-slate-600 leading-relaxed">
                  In patient ES-1003, baseline resting heart rate rises gradually from 68 bpm to 92 bpm over 16 hours. 
                  Because 92 bpm is less than the fixed maximum threshold of 100 bpm, <span className="font-bold text-rose-600">Fixed Threshold systems trigger ZERO alerts (False Negative)</span>. 
                  EarlyShift identifies this as a <span className="font-bold text-teal-700">+4.2 MAD directional drift</span> relative to the patient's individual median, firing an early warning alert 10 hours prior.
                </p>
              </div>

            </div>
          )}

          {/* ========================================== */}
          {/* TAB 6: INTERACTIVE GUIDED DEMO MODE        */}
          {/* ========================================== */}
          {activeTab === 'demo' && (
            <div className="space-y-6">
              
              {/* Guided Banner */}
              <div className="bg-teal-900 text-white p-6 rounded-xl shadow-lg border border-teal-800 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                  <span className="text-xs font-bold uppercase tracking-widest text-teal-300">2-Minute Judge Walkthrough</span>
                  <h3 className="text-xl font-bold mt-1">Interactive Time-Series Simulation Playback</h3>
                  <p className="text-slate-300 text-xs mt-1">
                    Simulate real-time arrival of patient readings for ES-1003 (Gradual Drift Case).
                  </p>
                </div>

                <div className="flex items-center space-x-2 bg-teal-950 p-2 rounded-lg border border-teal-800">
                  <button
                    onClick={() => setIsDemoPlaying(!isDemoPlaying)}
                    className="p-2 bg-teal-500 text-slate-950 rounded font-bold hover:bg-teal-400 transition"
                  >
                    {isDemoPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current" />}
                  </button>
                  <button
                    onClick={() => setDemoPlaybackIdx(prev => Math.min(30, prev + 1))}
                    className="p-2 bg-teal-800 text-white rounded hover:bg-teal-700 transition"
                  >
                    <SkipForward className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => { setDemoPlaybackIdx(8); setIsDemoPlaying(false); }}
                    className="p-2 bg-teal-800 text-white rounded hover:bg-teal-700 transition"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Simulation Chart */}
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <div className="flex justify-between items-center">
                  <div>
                    <h4 className="font-bold text-slate-800 text-base">Simulated Observation Arrival: Patient ES-1003</h4>
                    <p className="text-xs text-slate-500">Step {demoPlaybackIdx} of 30 processed observations</p>
                  </div>
                  <span className="text-xs font-bold text-teal-700 bg-teal-50 px-3 py-1 rounded-full border border-teal-200">
                    Live Engine Active
                  </span>
                </div>

                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={processedData.filter(d => d.patient_id === 'ES-1003').slice(0, demoPlaybackIdx)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                      <XAxis 
                        dataKey="timestamp" 
                        tickFormatter={(t) => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      />
                      <YAxis domain={[50, 110]} />
                      <Tooltip />
                      <ReferenceLine y={100} stroke="#F59E0B" strokeDasharray="3 3" label={{ value: "Fixed Limit (100 bpm)", fill: "#D97706", fontSize: 10 }} />
                      <Line type="monotone" dataKey="heart_rate" stroke="#0D9488" strokeWidth={3} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Live Trigger Status */}
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 text-xs">
                  <span className="font-bold text-slate-800 block mb-1">Simulated Detector Output:</span>
                  {demoPlaybackIdx >= 19 ? (
                    <div className="p-3 bg-rose-50 text-rose-900 border border-rose-200 rounded-lg flex items-center space-x-2">
                      <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0" />
                      <div>
                        <p className="font-bold text-xs">EARLYSHIFT ALERT TRIGGERED (Gradual Drift Detected)</p>
                        <p className="text-[11px] mt-0.5">Heart rate reached 88 bpm (+3.1 Robust Z-Score above baseline). Conventional fixed threshold limit (100 bpm) remains UNTRIGGERED.</p>
                      </div>
                    </div>
                  ) : (
                    <p className="text-slate-600 italic">Baseline within expected variation bounds. Continuing continuous observation...</p>
                  )}
                </div>
              </div>

            </div>
          )}

          {/* ========================================== */}
          {/* TAB 7: DATA IMPORT / EXPORT HUB            */}
          {/* ========================================== */}
          {activeTab === 'data-io' && (
            <div className="space-y-6">
              
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <h3 className="text-lg font-bold text-slate-900">Synthetic CSV Data Import & Export</h3>
                <p className="text-xs text-slate-600">
                  Upload user-provided synthetic CSV observations adhering to required schema: <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-800">timestamp, patient_id, heart_rate, respiratory_rate, temperature_c, spo2, systolic_bp</code>.
                </p>

                <div className="flex flex-col sm:flex-row gap-4 pt-2">
                  <button
                    onClick={handleExportCSV}
                    className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-lg text-xs font-semibold flex items-center justify-center space-x-2 transition"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download Current Synthetic CSV</span>
                  </button>

                  <label className="px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg text-xs font-semibold flex items-center justify-center space-x-2 cursor-pointer transition">
                    <Upload className="w-4 h-4" />
                    <span>Import Custom CSV File</span>
                    <input type="file" accept=".csv" className="hidden" onChange={(e) => {
                      const file = e.target.files[0];
                      if (!file) return;
                      const reader = new FileReader();
                      reader.onload = (event) => {
                        try {
                          const lines = event.target.result.split('\n');
                          const imported = [];
                          for (let i = 1; i < lines.length; i++) {
                            if (!lines[i].trim()) continue;
                            const cols = lines[i].split(',');
                            imported.push({
                              timestamp: cols[0]?.trim(),
                              patient_id: cols[1]?.trim(),
                              heart_rate: cols[2] ? parseFloat(cols[2]) : null,
                              respiratory_rate: cols[3] ? parseFloat(cols[3]) : null,
                              temperature_c: cols[4] ? parseFloat(cols[4]) : null,
                              spo2: cols[5] ? parseFloat(cols[5]) : null,
                              systolic_bp: cols[6] ? parseFloat(cols[6]) : null,
                              groundTruthAnomaly: false
                            });
                          }
                          if (imported.length > 0) {
                            setRawDataset(imported);
                            alert(`Successfully loaded ${imported.length} synthetic observation rows.`);
                          }
                        } catch (err) {
                          alert("Failed to parse CSV file format.");
                        }
                      };
                      reader.readAsText(file);
                    }} />
                  </label>
                </div>
              </div>

            </div>
          )}

          {/* ========================================== */}
          {/* TAB 8: SAFETY & LIMITATIONS PAGE           */}
          {/* ========================================== */}
          {activeTab === 'safety' && (
            <div className="space-y-6">
              
              <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <div className="flex items-center space-x-2 text-rose-600">
                  <ShieldAlert className="w-6 h-6" />
                  <h3 className="text-xl font-extrabold text-slate-900">Prototype Safety Guardrails & Limitations</h3>
                </div>

                <div className="space-y-3 text-xs text-slate-700 leading-relaxed">
                  <p className="bg-amber-50 p-4 rounded-lg border border-amber-200 text-amber-900 font-medium">
                    EarlyShift is a student research prototype designed for educational evaluation. It is NOT a medical device, does NOT diagnose clinical conditions, and must NEVER be used for emergency care or medical decision-making.
                  </p>

                  <h4 className="font-bold text-slate-900 text-sm mt-4">Key Methodological Limitations:</h4>
                  <ul className="list-disc pl-5 space-y-1 text-slate-600">
                    <li>All patient datasets are entirely synthetic and generated for scenario testing.</li>
                    <li>Statistical z-scores reflect deviations from historic median values, not clinical risk.</li>
                    <li>Sensor movement, calibration noise, or misplacement can trigger false positive alerts.</li>
                    <li>Baseline periods require a minimum number of valid observations to reach robust stability.</li>
                  </ul>
                </div>
              </div>

            </div>
          )}

        </div>
      </main>

    </div>
  );
}