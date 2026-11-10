using System;
using UnityEngine;

[Serializable]
public class ConversationMessage
{
    public int version = 1;
    public string type;
    public string turn_id;
    public string text;
    public string role;
    public string state;
    public bool demo;
    public int sequence;
    public int chunks;
    public int sample_rate;
    public int sample_count;
    public string audio_pcm16;
    public string[] joint_names;
    public double[] timestamps;
    public float[] angles;
    public double backend_elapsed_ms;
    public double gemini_ready_ms;
    public double gemini_gap_ms;
    public double motion_queue_ms;
    public double motion_inference_ms;
}

public static class ConversationProtocol
{
    public const int MaxMessageBytes = 4 * 1024 * 1024;
    public static readonly string[] JointNames = { "BodyYaw", "BodyPitch", "RightArmPitch", "RightArmRoll",
        "LeftArmPitch", "LeftArmRoll", "FacePitch", "FaceYaw", "FaceRoll", "EyePitch", "RightEyeYaw",
        "LeftEyeYaw", "Eyelid", "Mouth" };

    public static bool ValidateChunk(ConversationMessage message)
    {
        if (message == null || message.version != 1 || message.type != "response_chunk" ||
            string.IsNullOrEmpty(message.turn_id) || message.sequence < 0 ||
            message.sample_rate < 8000 || message.sample_rate > 96000 || message.sample_count < 1 ||
            message.sample_count > message.sample_rate * 30 || message.timestamps == null ||
            message.timestamps.Length < 1 || message.timestamps.Length > 2000 || message.angles == null ||
            message.angles.Length != message.timestamps.Length * 14 || message.joint_names == null ||
            message.joint_names.Length != 14 || string.IsNullOrEmpty(message.audio_pcm16)) return false;
        for (int j = 0; j < 14; j++) if (message.joint_names[j] != JointNames[j]) return false;
        for (int i = 0; i < message.timestamps.Length; i++)
            if (double.IsNaN(message.timestamps[i]) || double.IsInfinity(message.timestamps[i]) ||
                message.timestamps[i] < 0 || message.timestamps[i] >= (double)message.sample_count / message.sample_rate + .0001 ||
                (i > 0 && message.timestamps[i] <= message.timestamps[i - 1])) return false;
        foreach (float value in message.angles) if (float.IsNaN(value) || float.IsInfinity(value)) return false;
        return true;
    }

    public static float[] DecodeAudio(ConversationMessage message)
    {
        byte[] bytes = Convert.FromBase64String(message.audio_pcm16);
        if (bytes.Length != message.sample_count * 2) throw new FormatException("PCM length does not match sample_count");
        var samples = new float[message.sample_count];
        for (int i = 0; i < samples.Length; i++) samples[i] = (short)(bytes[2 * i] | bytes[2 * i + 1] << 8) / 32768f;
        return samples;
    }

    public static string EncodeAudio(float[] samples)
    {
        var bytes = new byte[samples.Length * 2];
        for (int i = 0; i < samples.Length; i++)
        {
            short value = (short)Mathf.RoundToInt(Mathf.Clamp(samples[i], -1, 1) * 32767);
            bytes[2 * i] = (byte)(value & 255);
            bytes[2 * i + 1] = (byte)((value >> 8) & 255);
        }
        return Convert.ToBase64String(bytes);
    }

    public static void Interpolate(ConversationMessage message, double seconds, float[] result)
    {
        int upper = Array.BinarySearch(message.timestamps, seconds);
        if (upper < 0) upper = ~upper;
        upper = Math.Min(upper, message.timestamps.Length - 1);
        int lower = Math.Max(upper - 1, 0);
        double span = message.timestamps[upper] - message.timestamps[lower];
        float fraction = span > 0 ? Mathf.Clamp01((float)((seconds - message.timestamps[lower]) / span)) : 0;
        for (int j = 0; j < 14; j++) result[j] = Mathf.Lerp(message.angles[lower * 14 + j], message.angles[upper * 14 + j], fraction);
    }
}
