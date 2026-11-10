using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class ConversationClient : MonoBehaviour
{
    public string backendUrl = "ws://127.0.0.1:8765";
    public ConversationState state;
    public SpeechMotionPlayer player;
    public MicrophoneCapture microphone;
    public bool connectOnStart = true;
    ClientWebSocket socket;
    CancellationTokenSource lifetime;
    readonly SemaphoreSlim sendLock = new SemaphoreSlim(1, 1);
    readonly ConcurrentQueue<Action> mainThread = new ConcurrentQueue<Action>();
    int connectionGeneration, expectedSequence, expectedChunks = -1;
    bool completed;
    int pendingSends;

    void Start()
    {
        microphone.Started += AudioStarted;
        microphone.Chunk += SendAudio;
        microphone.Ended += AudioEnded;
        microphone.Failed += Fail;
        if (connectOnStart) Connect();
    }

    public async void Connect()
    {
        Disconnect();
        int generation = connectionGeneration;
        state.status = "Connecting";
        lifetime = new CancellationTokenSource();
        var token = lifetime.Token;
        var connection = new ClientWebSocket();
        socket = connection;
        try
        {
            await connection.ConnectAsync(new Uri(backendUrl), token);
            await Receive(connection, generation, token);
        }
        catch (OperationCanceledException) { }
        catch (Exception error) { mainThread.Enqueue(() => { if (generation == connectionGeneration) Fail(error.Message); }); }
        finally
        {
            connection.Dispose();
            mainThread.Enqueue(() => { if (generation == connectionGeneration) { state.connected = false; player.StopAll(); microphone.Cancel(); if (!state.status.StartsWith("Error:")) state.status = "Disconnected"; } });
        }
    }

    async Task Receive(ClientWebSocket connection, int generation, CancellationToken token)
    {
        var buffer = new byte[16384];
        while (connection.State == WebSocketState.Open && !token.IsCancellationRequested)
        {
            using (var data = new MemoryStream())
            {
                WebSocketReceiveResult result;
                do
                {
                    result = await connection.ReceiveAsync(new ArraySegment<byte>(buffer), token).ConfigureAwait(false);
                    if (result.MessageType == WebSocketMessageType.Close) return;
                    if (result.MessageType != WebSocketMessageType.Text) throw new FormatException("Expected JSON text");
                    data.Write(buffer, 0, result.Count);
                    if (data.Length > ConversationProtocol.MaxMessageBytes) throw new FormatException("Response too large");
                } while (!result.EndOfMessage);
                string json = Encoding.UTF8.GetString(data.ToArray());
                if (mainThread.Count >= 32) throw new InvalidOperationException("Incoming response queue is full");
                mainThread.Enqueue(() => { if (generation == connectionGeneration) Handle(json); });
            }
        }
    }

    async void Send(ConversationMessage message)
    {
        var connection = socket;
        int generation = connectionGeneration;
        var token = lifetime != null ? lifetime.Token : CancellationToken.None;
        if (connection == null || connection.State != WebSocketState.Open) { Fail("Connect to the backend first"); return; }
        try
        {
            if (Interlocked.Increment(ref pendingSends) > 32)
            { Interlocked.Decrement(ref pendingSends); throw new InvalidOperationException("Outgoing audio queue is full; reconnect"); }
            try
            {
                byte[] data = Encoding.UTF8.GetBytes(JsonUtility.ToJson(message));
                await sendLock.WaitAsync(token);
                try
                {
                    if (generation != connectionGeneration) return;
                    if (message.type != "cancel" && message.type != "playback_done" && message.turn_id != state.activeTurn) return;
                    await connection.SendAsync(new ArraySegment<byte>(data), WebSocketMessageType.Text, true, token);
                }
                finally { sendLock.Release(); }
            }
            finally { Interlocked.Decrement(ref pendingSends); }
        }
        catch (OperationCanceledException) { }
        catch (Exception error) { if (generation == connectionGeneration) Fail(error.Message); }
    }

    void Handle(string json)
    {
        try
        {
            var message = JsonUtility.FromJson<ConversationMessage>(json);
            if (message == null || message.version != 1) throw new FormatException("Invalid server message");
            if (message.type == "ready")
            { state.connected = true; state.demo = message.demo; state.status = "Listening"; return; }
            if (message.type == "error" && string.IsNullOrEmpty(message.turn_id)) { Fail(message.text); return; }
            if (message.turn_id != state.activeTurn) return; // Stale generation/cancelled turn.
            switch (message.type)
            {
                case "cancelled":
                    CancelTurn(); break;
                case "response_chunk":
                    if (message.sequence != expectedSequence) throw new FormatException("Missing/duplicate speech phrase");
                    player.Enqueue(message); expectedSequence++; state.status = "Buffering"; break;
                case "response_complete":
                    expectedChunks = message.chunks;
                    if (expectedChunks != expectedSequence) throw new FormatException("Incomplete speech response");
                    player.CompleteResponse(); completed = true; break;
                case "transcript":
                    if (message.role == "user") state.userTranscript = message.text;
                    else state.robotTranscript = message.text;
                    break;
                case "state":
                    if (!player.IsBusy) state.status = message.state;
                    break;
                case "error": Fail(message.text); break;
            }
        }
        catch (Exception error) { CancelTurn(); Fail(error.Message); }
    }

    void Update()
    {
        int count = 0;
        while (count++ < 32 && mainThread.TryDequeue(out var action)) action();
        if (player.IsBusy)
            state.status = player.IsBuffering ? (player.IsRebuffering ? "Buffering more speech" : "Buffering speech") : "Speaking";
        if (completed && expectedSequence == expectedChunks && !player.IsBusy)
        {
            completed = false;
            Send(new ConversationMessage { type = "playback_done", turn_id = state.activeTurn });
            state.status = "Listening";
        }
    }

    void BeginTurn()
    {
        CancelTurn();
        state.NewTurn(); expectedSequence = 0; expectedChunks = -1; completed = false;
        state.robotTranscript = ""; state.status = "Thinking";
    }
    public void SendText(string text)
    {
        if (!state.connected || string.IsNullOrWhiteSpace(text)) return;
        microphone.Cancel(); BeginTurn();
        Send(new ConversationMessage { type = "text", turn_id = state.activeTurn, text = text });
    }
    public void Preview()
    {
        if (!state.connected) return;
        microphone.Cancel(); BeginTurn();
        Send(new ConversationMessage { type = "demo", turn_id = state.activeTurn });
    }
    public void BeginRecording()
    {
        if (!state.connected || state.demo) return;
        BeginTurn(); state.status = "Recording"; microphone.Begin();
    }
    void AudioStarted()
    {
        if (state.connected && !string.IsNullOrEmpty(state.activeTurn))
            Send(new ConversationMessage { type = "audio_start", turn_id = state.activeTurn });
    }
    void SendAudio(float[] audio, int rate)
    {
        if (!state.connected || string.IsNullOrEmpty(state.activeTurn)) return;
        Send(new ConversationMessage { type = "audio_chunk", turn_id = state.activeTurn,
            sample_rate = rate, audio_pcm16 = ConversationProtocol.EncodeAudio(audio) });
    }
    void AudioEnded()
    {
        if (!state.connected || string.IsNullOrEmpty(state.activeTurn)) return;
        Send(new ConversationMessage { type = "audio_end", turn_id = state.activeTurn });
        state.status = "Thinking";
    }
    public void CancelTurn()
    {
        string turn = state.activeTurn;
        state.activeTurn = null; completed = false; expectedSequence = 0;
        if (!string.IsNullOrEmpty(turn) && socket != null && socket.State == WebSocketState.Open)
            Send(new ConversationMessage { type = "cancel", turn_id = turn });
        player.StopAll(); microphone.Cancel();
        state.status = state.connected ? "Listening" : "Disconnected";
    }
    void Fail(string message)
    {
        CancelTurn(); state.status = "Error: " + message;
    }
    public void Disconnect()
    {
        connectionGeneration++;
        lifetime?.Cancel(); socket?.Abort(); socket?.Dispose(); lifetime?.Dispose();
        socket = null; lifetime = null;
        if (player != null) player.StopAll();
        if (microphone != null) microphone.Cancel();
        if (state != null) { state.connected = false; state.activeTurn = null; state.status = "Disconnected"; }
    }
    void OnDisable() { Disconnect(); }
    void OnDestroy()
    {
        if (microphone != null) { microphone.Started -= AudioStarted; microphone.Chunk -= SendAudio; microphone.Ended -= AudioEnded; microphone.Failed -= Fail; }
    }
}
