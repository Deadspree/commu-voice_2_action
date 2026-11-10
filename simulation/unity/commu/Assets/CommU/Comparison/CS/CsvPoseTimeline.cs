using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

// DINOv3 inference CSVs contain absolute degrees and video-relative timestamps.
public sealed class CsvPoseTimeline
{
    public static readonly string[] JointNames = { "BodyYaw", "BodyPitch", "RightArmPitch", "RightArmRoll",
        "LeftArmPitch", "LeftArmRoll", "FacePitch", "FaceYaw", "FaceRoll", "EyePitch", "RightEyeYaw",
        "LeftEyeYaw", "Eyelid", "Mouth" };
    public readonly double[] Times;
    public readonly float[][] Angles;
    public int Count => Times.Length;
    public double Duration => Times[Count - 1] + (Count > 1 ? Times[Count - 1] - Times[Count - 2] : 1.0 / 30);

    public CsvPoseTimeline(string path)
    {
        using (var reader = new StreamReader(path))
        {
            string[] header = (reader.ReadLine() ?? "").Trim().Split(',');
            if (header.Length != 16 || header[0] != "frame_index" || header[1] != "timestamp_seconds")
                throw new InvalidDataException("Expected a timestamped pose-estimation CSV");
            for (int j = 0; j < 14; j++)
                if (header[j + 2] != JointNames[j]) throw new InvalidDataException("CSV joint order does not match CommU");
            var times = new List<double>();
            var angles = new List<float[]>();
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                string[] columns = line.Split(',');
                if (columns.Length != 16 || !int.TryParse(columns[0], out int index) || index != times.Count)
                    throw new InvalidDataException("Invalid CSV frame index");
                double time = double.Parse(columns[1], CultureInfo.InvariantCulture);
                if (double.IsNaN(time) || double.IsInfinity(time) || time < 0 ||
                    (times.Count > 0 && time <= times[times.Count - 1])) throw new InvalidDataException("Invalid CSV timestamps");
                var values = new float[14];
                for (int j = 0; j < 14; j++)
                {
                    values[j] = float.Parse(columns[j + 2], CultureInfo.InvariantCulture);
                    if (float.IsNaN(values[j]) || float.IsInfinity(values[j])) throw new InvalidDataException("Nonfinite joint angle");
                }
                times.Add(time); angles.Add(values);
            }
            if (times.Count == 0 || times[0] > .001) throw new InvalidDataException("CSV must start at time zero and contain poses");
            Times = times.ToArray(); Angles = angles.ToArray();
        }
    }

    public void Sample(double seconds, float[] output)
    {
        int upper = Array.BinarySearch(Times, seconds);
        if (upper < 0) upper = ~upper;
        upper = Math.Min(upper, Count - 1);
        int lower = Math.Max(0, upper - 1);
        double span = Times[upper] - Times[lower];
        float fraction = span > 0 ? Mathf.Clamp01((float)((seconds - Times[lower]) / span)) : 0;
        for (int j = 0; j < 14; j++) output[j] = Mathf.Lerp(Angles[lower][j], Angles[upper][j], fraction);
    }
}

[Serializable] public class ComparisonRunManifest
{
    [Serializable] public class FileInput { public string path; }
    [Serializable] public class Inputs { public FileInput video; }
    [Serializable] public class Timeline { public double source_start_seconds; }
    public string status;
    public Inputs inputs;
    public Timeline timeline;
}

public sealed class ComparisonRun
{
    public readonly string Folder, Original, Generated, Csv;
    public readonly double OriginalOffset;
    public readonly CsvPoseTimeline Poses;
    public static string RepositoryRoot => Path.GetFullPath(Path.Combine(Application.dataPath, "../../../.."));

    public ComparisonRun(string folder)
    {
        Folder = Path.GetFullPath(Path.IsPathRooted(folder) ? folder : Path.Combine(RepositoryRoot, folder));
        var manifest = JsonUtility.FromJson<ComparisonRunManifest>(File.ReadAllText(Path.Combine(Folder, "manifest.json")));
        if (manifest.status != "completed") throw new InvalidDataException("Select a completed Lucy run");
        string source = manifest.inputs?.video?.path;
        if (string.IsNullOrEmpty(source)) throw new InvalidDataException("Manifest has no original video");
        if (!File.Exists(source))
        {
            int dataset = source.Replace('\\', '/').IndexOf("/dataset/", StringComparison.Ordinal);
            if (dataset >= 0) source = Path.Combine(RepositoryRoot, source.Substring(dataset + 1));
        }
        Original = Path.GetFullPath(source);
        Generated = Path.Combine(Folder, "robot_with_audio.mp4");
        Csv = Path.Combine(Folder, "robot_with_audio_joints.csv");
        foreach (string file in new[] { Original, Generated, Csv })
            if (!File.Exists(file)) throw new FileNotFoundException("Comparison input is missing", file);
        OriginalOffset = manifest.timeline?.source_start_seconds ?? 0;
        if (OriginalOffset < 0 || double.IsNaN(OriginalOffset) || double.IsInfinity(OriginalOffset))
            throw new InvalidDataException("Invalid source offset");
        Poses = new CsvPoseTimeline(Csv);
    }
}
