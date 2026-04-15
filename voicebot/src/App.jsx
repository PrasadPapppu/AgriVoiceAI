import { useState, useRef, useEffect } from "react";
import { FaMicrophone, FaStop } from "react-icons/fa";

function App() {
  const [messages, setMessages] = useState([]);
  const [recording, setRecording] = useState(false);
  const [liveText, setLiveText] = useState(""); // 🔥 STREAMING TEXT

  const wsRef = useRef(null);
  const processorRef = useRef(null);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);

  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);

  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, liveText]);

  // =============================
  // 🔊 AUDIO PLAYER
  // =============================
  const playNextAudio = () => {
    if (audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }

    isPlayingRef.current = true;

    const base64 = audioQueueRef.current.shift();
    const audio = new Audio("data:audio/wav;base64," + base64);

    window.currentAudio = audio;

    audio.onended = playNextAudio;
    audio.onerror = playNextAudio;

    audio.play().catch(playNextAudio);
  };

  const stopAudio = () => {
    audioQueueRef.current = [];
    isPlayingRef.current = false;

    if (window.currentAudio) {
      window.currentAudio.pause();
      window.currentAudio.src = "";
      window.currentAudio = null;
    }
  };

  // =============================
  // 🎤 START RECORDING
  // =============================
  const startRecording = async () => {
    try {
      if (audioContextRef.current) {
        await audioContextRef.current.close();
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: 16000 });
      await audioContext.resume();
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      processorRef.current = processor;

      source.connect(processor);
      processor.connect(audioContext.destination);

      const ws = new WebSocket("ws://localhost:8000/ws/audio");
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => console.log("✅ WS CONNECTED");
      ws.onclose = () => console.log("❌ WS CLOSED");
      ws.onerror = (e) => console.log("⚠️ WS ERROR", e);

      // =============================
      // 📩 MESSAGE HANDLER (STREAMING)
      // =============================
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        // 🧑 USER MESSAGE
        if (data.type === "user") {
          setMessages((prev) => [
            ...prev,
            { role: "user", text: data.text }
          ]);
        }

        // 🔥 STREAM TOKENS
        else if (data.type === "stream") {
          setLiveText((prev) => prev + data.token);
        }

        // ✅ FINAL RESPONSE
        else if (data.type === "final") {
          setMessages((prev) => [
            ...prev,
            { role: "ai", text: data.text }
          ]);

          setLiveText(""); // clear streaming buffer
        }

        // 🔊 AUDIO QUEUE
        else if (data.type === "audio") {
          audioQueueRef.current.push(data.audio);
          if (!isPlayingRef.current) playNextAudio();
        }
      };

      // =============================
      // 🎤 AUDIO STREAM
      // =============================
      processor.onaudioprocess = (e) => {
        if (
          !wsRef.current ||
          wsRef.current.readyState !== WebSocket.OPEN ||
          wsRef.current.bufferedAmount > 300000
        ) return;

        const input = e.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(input.length);

        for (let i = 0; i < input.length; i++) {
          int16[i] = Math.max(-1, Math.min(1, input[i])) * 32767;
        }

        wsRef.current.send(int16.buffer);
      };

      setRecording(true);

    } catch (err) {
      console.error("Mic error:", err);
    }
  };

  // =============================
  // 🛑 STOP RECORDING
  // =============================
  const stopRecording = () => {
    stopAudio();

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setRecording(false);
  };

  // =============================
  // UI
  // =============================
  return (
    <div style={{
      height: "100vh",
      background: "#020617",
      color: "white",
      display: "flex",
      flexDirection: "column"
    }}>

      <div style={{
        padding: "14px 24px",
        borderBottom: "1px solid rgba(255,255,255,0.06)"
      }}>
        AI Voice Assistant (Streaming)
      </div>

      <div style={{
        flex: 1,
        overflowY: "auto",
        padding: "20px"
      }}>
        {messages.map((m, i) => (
          <div key={i} style={{
            textAlign: m.role === "user" ? "right" : "left",
            marginBottom: "12px",
            maxWidth: "70%",
            marginLeft: m.role === "user" ? "auto" : "0",
            wordWrap: "break-word",
            lineHeight: "1.5"
          }}>
            {m.text}
          </div>
        ))}

        {/* 🔥 LIVE STREAMING TEXT */}
        {liveText && (
          <div style={{
            textAlign: "left",
            marginBottom: "12px",
            maxWidth: "70%",
            opacity: 0.8
          }}>
            {liveText}
            <span style={{ marginLeft: 4 }}>▍</span>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      <div style={{
        padding: "20px",
        display: "flex",
        justifyContent: "center"
      }}>
        {!recording ? (
          <button onClick={startRecording}>
            <FaMicrophone />
          </button>
        ) : (
          <button onClick={stopRecording}>
            <FaStop />
          </button>
        )}
      </div>

    </div>
  );
}

export default App;
