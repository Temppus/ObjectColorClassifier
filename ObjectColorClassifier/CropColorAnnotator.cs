using OpenCvSharp;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;

namespace ObjectColorClassifier
{
    internal class CropColorAnnotator
    {
        private readonly ColorClassifierLib.ObjectColorClassifier _classifier;

        public CropColorAnnotator(ColorClassifierLib.ObjectColorClassifier classifier)
        {
            _classifier = classifier ?? throw new ArgumentNullException(nameof(classifier));
        }

        public void ClassifyCropsInDirectory(string sourceDir, int? downscaleToMaxSize = null)
        {
            var fullPathSourceDir = Path.GetFullPath(sourceDir);

            if (!Directory.Exists(fullPathSourceDir))
            {
                throw new DirectoryNotFoundException($"Directory {fullPathSourceDir} not found");
            }

            var sourceDirName = Path.GetFileName(fullPathSourceDir);
            var parentDir = Path.GetDirectoryName(fullPathSourceDir);
            var resultDir = Path.Combine(parentDir, $"{sourceDirName}_Annotated");

            Directory.CreateDirectory(resultDir);

            foreach (var sourceFile in Directory.GetFiles(fullPathSourceDir).OrderBy(File.GetCreationTime))
            {
                using var origSourceMat = Cv2.ImRead(sourceFile);
                using var sourceMatResized = ResizeToMax(origSourceMat, downscaleToMaxSize);

                var sw = Stopwatch.StartNew();
                var baseColor = _classifier.Classify(sourceMatResized, out var grabCut, out var x, out var y);
                sw.Stop();

                Console.WriteLine($"Color is {baseColor}. Took {sw.ElapsedMilliseconds}ms");

                var id = Guid.NewGuid().ToString()[..8];

                {
                    string prefix = $"{id}-{baseColor}-";
                    var dstFilePath = GetResultFilePath(resultDir, prefix, sourceFile);
                    sourceMatResized.SaveImage(dstFilePath);
                }

                {
                    string prefix = $"{id}-{baseColor}-GrabCut-";
                    var dstFilePath = GetResultFilePath(resultDir, prefix, sourceFile);
                    grabCut.SaveImage(dstFilePath);
                }

                {
                    string prefix = $"{id}-{baseColor}-X";
                    var dstFilePath = GetResultFilePath(resultDir, prefix, sourceFile);
                    x.SaveImage(dstFilePath);
                }

                {
                    string prefix = $"{id}-{baseColor}-Y";
                    var dstFilePath = GetResultFilePath(resultDir, prefix, sourceFile);
                    y.SaveImage(dstFilePath);
                }
            }
        }

        private static string GetResultFilePath(string resultDir, string prefix, string sourceFile)
        {
            var sourceFileName = Path.GetFileName(sourceFile);
            var resultFileName = $"{prefix}_{sourceFileName}";
            return Path.Combine(resultDir, resultFileName);
        }

        private static Mat ResizeToMax(Mat sourceMat, int? maxSize)
        {
            if (maxSize == null)
            {
                return sourceMat.Clone();
            }

            int widthRatio = sourceMat.Width / maxSize.Value;
            int heightRatio = sourceMat.Height / maxSize.Value;

            var sourceMatResized = new Mat();

            if (widthRatio > 1 || heightRatio > 1)
            {
                var ratio = widthRatio > heightRatio ? widthRatio : heightRatio;
                var newSize = new Size(sourceMat.Width / ratio, sourceMat.Height / ratio);
                Cv2.Resize(sourceMat, sourceMatResized, newSize);
                return sourceMatResized;
            }

            return sourceMat.Clone();
        }
    }
}
