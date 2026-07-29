using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

public abstract class DataCollectorBase : MonoBehaviour
{
	[SerializeField] protected Camera captureCamera;
	[SerializeField] protected CS_TestRotateBones poseSource;
	[SerializeField] protected bool captureOnStart = false;
	[SerializeField] protected float captureIntervalSeconds = 0f;
	[SerializeField] protected string datasetFolderName = "dataset/commu_pose_dataset";
	[SerializeField] protected string imagesFolderName = "images";
	[SerializeField] protected string csvFileName = "angle_joints.csv";

	protected string DatasetRootPath { get; private set; }
	protected string ImagesFolderPath { get; private set; }
	protected string CsvFilePath { get; private set; }
	protected int NextSampleIndex { get; private set; }

	protected virtual void Awake()
	{
		if (captureCamera == null)
		{
			captureCamera = Camera.main;
		}

		if (captureCamera == null)
		{
			captureCamera = FindObjectOfType<Camera>();
		}

		if (poseSource == null)
		{
			poseSource = FindObjectOfType<CS_TestRotateBones>();
		}
	}

	protected virtual void Start()
	{
		EnsureStoragePaths();
		NextSampleIndex = GetNextSampleIndex();
		Debug.Log("DataCollectorBase saving to: " + DatasetRootPath);

		if (captureOnStart)
		{
			CaptureSample();
		}

		if (captureIntervalSeconds > 0f)
		{
			StartCoroutine(CaptureLoop());
		}
	}

	public void CaptureSample()
	{
		StartCoroutine(CaptureSampleRoutine());
	}

	protected virtual IEnumerator CaptureLoop()
	{
		while (true)
		{
			CaptureSample();

			if (captureIntervalSeconds <= 0f)
			{
				yield break;
			}

			yield return new WaitForSeconds(captureIntervalSeconds);
		}
	}

	protected virtual IEnumerator CaptureSampleRoutine()
	{
		if (captureCamera == null)
		{
			Debug.LogWarning("DataCollectorBase: no capture camera is assigned.");
			yield break;
		}

		if (poseSource == null)
		{
			poseSource = FindObjectOfType<CS_TestRotateBones>();
		}

		if (poseSource == null)
		{
			Debug.LogWarning("DataCollectorBase: no pose source was found.");
			yield break;
		}

		int sampleIndex = NextSampleIndex;
		NextSampleIndex++;
		float[] jointAngles = ReadJointAngles();

		yield return new WaitForEndOfFrame();

		string imageFileName = sampleIndex.ToString() + ".png";
		string imagePath = Path.Combine(ImagesFolderPath, imageFileName);
		CaptureCameraFrame(captureCamera, imagePath);
		AppendCsvRow(sampleIndex, imageFileName, jointAngles);
		Debug.Log("DataCollectorBase captured sample " + sampleIndex + " -> " + imagePath);
	}

	protected virtual void EnsureStoragePaths()
	{
		string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "../../../../"));
		DatasetRootPath = Path.GetFullPath(Path.Combine(projectRoot, datasetFolderName));
		ImagesFolderPath = Path.Combine(DatasetRootPath, imagesFolderName);
		CsvFilePath = Path.Combine(DatasetRootPath, csvFileName);

		Directory.CreateDirectory(DatasetRootPath);
		Directory.CreateDirectory(ImagesFolderPath);
	}

	protected virtual int GetNextSampleIndex()
	{
		int maxIndex = -1;

		if (File.Exists(CsvFilePath))
		{
			string[] csvLines = File.ReadAllLines(CsvFilePath);
			for (int lineIndex = 1; lineIndex < csvLines.Length; lineIndex++)
			{
				string line = csvLines[lineIndex].Trim();
				if (string.IsNullOrEmpty(line))
				{
					continue;
				}

				string[] columns = line.Split(',');
				int parsedIndex;
				if (columns.Length > 0 && int.TryParse(columns[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out parsedIndex))
				{
					if (parsedIndex > maxIndex)
					{
						maxIndex = parsedIndex;
					}
				}
			}
		}

		if (Directory.Exists(ImagesFolderPath))
		{
			string[] imageFiles = Directory.GetFiles(ImagesFolderPath, "*.png");
			foreach (string imageFile in imageFiles)
			{
				string fileName = Path.GetFileNameWithoutExtension(imageFile);
				int parsedIndex;
				if (int.TryParse(fileName, NumberStyles.Integer, CultureInfo.InvariantCulture, out parsedIndex))
				{
					if (parsedIndex > maxIndex)
					{
						maxIndex = parsedIndex;
					}
				}
			}
		}

		return maxIndex + 1;
	}

	protected virtual float[] ReadJointAngles()
	{
		if (poseSource == null || poseSource.JointAngles == null)
		{
			return new float[0];
		}

		float[] sampledAngles = new float[poseSource.JointAngles.Length];
		Array.Copy(poseSource.JointAngles, sampledAngles, poseSource.JointAngles.Length);
		return sampledAngles;
	}

	protected virtual void CaptureCameraFrame(Camera cameraToCapture, string outputPath)
	{
		int width = Mathf.Max(1, cameraToCapture.pixelWidth);
		int height = Mathf.Max(1, cameraToCapture.pixelHeight);
		if (width <= 1 || height <= 1)
		{
			width = Mathf.Max(1, Screen.width);
			height = Mathf.Max(1, Screen.height);
		}

		RenderTexture renderTexture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
		RenderTexture previousActive = RenderTexture.active;
		RenderTexture previousTarget = cameraToCapture.targetTexture;

		try
		{
			cameraToCapture.targetTexture = renderTexture;
			cameraToCapture.Render();

			RenderTexture.active = renderTexture;
			Texture2D screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);
			screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
			screenshot.Apply();

			File.WriteAllBytes(outputPath, screenshot.EncodeToPNG());
			Destroy(screenshot);
		}
		finally
		{
			cameraToCapture.targetTexture = previousTarget;
			RenderTexture.active = previousActive;
			renderTexture.Release();
			Destroy(renderTexture);
		}
	}

	protected virtual void AppendCsvRow(int sampleIndex, string imageFileName, float[] jointAngles)
	{
		bool writeHeader = !File.Exists(CsvFilePath) || new FileInfo(CsvFilePath).Length == 0;
		using (StreamWriter writer = new StreamWriter(CsvFilePath, true, Encoding.UTF8))
		{
			if (writeHeader)
			{
				writer.WriteLine(BuildCsvHeader());
			}

			StringBuilder rowBuilder = new StringBuilder();
			rowBuilder.Append(sampleIndex.ToString(CultureInfo.InvariantCulture));
			rowBuilder.Append(',');
			rowBuilder.Append(imageFileName);

			for (int i = 0; i < jointAngles.Length; i++)
			{
				rowBuilder.Append(',');
				rowBuilder.Append(jointAngles[i].ToString("0.######", CultureInfo.InvariantCulture));
			}

			writer.WriteLine(rowBuilder.ToString());
		}
	}

	protected virtual string BuildCsvHeader()
	{
		string[] jointColumnNames = GetJointColumnNames();
		StringBuilder headerBuilder = new StringBuilder();
		headerBuilder.Append("index,image_file");

		for (int i = 0; i < jointColumnNames.Length; i++)
		{
			headerBuilder.Append(',');
			headerBuilder.Append(jointColumnNames[i]);
		}

		return headerBuilder.ToString();
	}

	protected abstract string[] GetJointColumnNames();
}