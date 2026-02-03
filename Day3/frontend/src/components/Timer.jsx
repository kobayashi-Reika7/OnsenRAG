/**
 * 1タスク用タイマー（開始・停止・リセット・経過表示）
 * 経過秒数はフロントで計算し、必要に応じて親経由で API に time を送る
 */
import React, { useState, useEffect, useRef } from 'react';

function formatElapsed(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function Timer({ taskId, initialTime, onTimeChange }) {
  const [elapsed, setElapsed] = useState(initialTime || 0);
  const [running, setRunning] = useState(false);
  const startRef = useRef(null);

  useEffect(() => {
    if (!running) return;
    const start = Date.now() - elapsed * 1000;
    startRef.current = start;
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [running]);

  const handleStart = () => setRunning(true);
  const handleStop = () => {
    setRunning(false);
    const total = elapsed;
    if (onTimeChange) onTimeChange(total);
  };
  const handleReset = () => {
    setRunning(false);
    setElapsed(0);
    if (onTimeChange) onTimeChange(0);
  };

  return (
    <div className="timer-row">
      <span className="timer-display">{formatElapsed(elapsed)}</span>
      <div className="timer-btns">
        <button type="button" className="timer-start" onClick={handleStart} disabled={running}>
          開始
        </button>
        <button type="button" className="timer-stop" onClick={handleStop} disabled={!running}>
          停止
        </button>
        <button type="button" className="timer-reset" onClick={handleReset}>
          リセット
        </button>
      </div>
    </div>
  );
}

export default Timer;
