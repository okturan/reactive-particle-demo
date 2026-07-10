function Hud() {
  return (
    <div className="hud" aria-live="polite">
      <div className="brand">
        <img id="brandMark" className="brand-logo" alt="" />
      </div>
      <div className="hud-actions">
        <div className="mode-switch" role="group" aria-label="Tracking mode">
          <button id="handModeButton" type="button" aria-pressed="true">
            HAND
          </button>
          <button id="faceModeButton" type="button" aria-pressed="false">
            FACE
          </button>
        </div>
        <select id="cameraSelect" className="camera-select" aria-label="Camera" />
        <button id="cameraRefreshButton" className="camera-refresh" type="button" aria-label="Rescan cameras">
          SCAN
        </button>
        <button id="pointerToggle" className="pointer-toggle" type="button" aria-pressed="false">
          MOUSE OFF
        </button>
        <button id="debugToggle" className="pointer-toggle" type="button" aria-pressed="false">
          DEBUG OFF
        </button>
        <button id="calibrationToggle" className="pointer-toggle" type="button" aria-pressed="false">
          TUNE OFF
        </button>
        <button id="settingsToggle" className="pointer-toggle" type="button" aria-pressed="false">
          SETTINGS
        </button>
        <div id="perfStatus" className="status perf-status">
          -- FPS
        </div>
        <div id="cameraStatus" className="status">
          CAMERA
        </div>
      </div>
    </div>
  );
}

function TrackingDebug() {
  return (
    <div id="handDebug" className="hand-debug" aria-label="Camera tracking debug">
      <div className="debug-head">
        <span>TRACKING</span>
        <span id="handDebugMeta">OFF</span>
      </div>
      <div className="debug-stage">
        <video id="handVideo" playsInline muted />
        <canvas id="handOverlay" />
      </div>
      <div className="debug-readout">
        <span id="handDebugHands">0 HANDS</span>
        <span id="handDebugFps">0 FPS</span>
        <span id="handDebugGesture">OPEN 0.00</span>
      </div>
    </div>
  );
}

function CalibrationPanel() {
  return (
    <div id="calibrationPanel" className="calibration-panel" aria-label="Face calibration controls" hidden>
      <div className="calibration-head">
        <span>FACE TUNE</span>
        <button id="calibrationReset" type="button">
          RESET
        </button>
      </div>
      <div id="calibrationControls" className="calibration-controls" />
      <textarea id="calibrationValues" readOnly aria-label="Calibration values" />
    </div>
  );
}

function SettingsPanel() {
  return (
    <div id="settingsPanel" className="settings-panel" aria-label="Particle and hand settings" hidden>
      <div className="settings-head">
        <span>PARTICLE SETTINGS</span>
        <button id="settingsReset" type="button">
          RESET
        </button>
      </div>
      <div id="settingsControls" className="settings-controls" />
      <textarea id="settingsValues" readOnly aria-label="Particle setting values" />
    </div>
  );
}

export default function App() {
  return (
    <div id="app">
      <video id="faceVideoBackdrop" className="face-video-backdrop" playsInline muted aria-hidden="true" />
      <canvas id="scene" aria-label="Reactive particle field" />
      <Hud />
      <div id="hintBar" className="hint-bar" role="status" title="Click to dismiss" hidden />
      <TrackingDebug />
      <CalibrationPanel />
      <SettingsPanel />
    </div>
  );
}
