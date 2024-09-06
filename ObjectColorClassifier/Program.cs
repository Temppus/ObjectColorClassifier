using System.Diagnostics;
using ColorClassifierLib;
using OpenCvSharp;

namespace ObjectColorClassifier
{
    internal class Program
    {
        static void Main(string[] args)
        {
            const ulong seed = 420;
            Cv2.SetTheRNG(seed);

            const string sourceDir = "C:\\Users\\ibenovic\\Desktop\\SFCrops\\PedestrianCrops";
            IBaseColorClassifier baseColorClassifier;

            if (true)
            {
                baseColorClassifier = new NeuralBaseColorClassifier(@"C:\Users\ibenovic\Desktop\rgb_color_classifier.onnx");
            }
            else
            {
                baseColorClassifier = new SimpleDistanceBaseColorClassifier();
            }

            var objColorClassifier = new ColorClassifierLib.ObjectColorClassifier(baseColorClassifier)
            {
                ReduceFocusArea = true
            };

            var annotator = new CropColorAnnotator(objColorClassifier);

            var sw = Stopwatch.StartNew();
            annotator.ClassifyCropsInDirectory(sourceDir, downscaleToMaxSize: 200);
            sw.Stop();

            Console.WriteLine($"Totally took {sw.ElapsedMilliseconds}ms");

            /*
            // Load the image
            using var origSourceMat = Cv2.ImRead(@"C:\Users\ibenovic\Desktop\person_cropped_10.jpg");

            const int maxSize = 200;
            using var sourceMatResized = ResizeToMax(origSourceMat, maxSize);

            // warmup
            using var classifier = new ColorClassifierLib.ObjectColorClassifier(modelPath);
            var colorConfidenceMap = classifier.Classify(sourceMatResized, out var grabCut, out var x, out var y);

            // measure time
            var sw = Stopwatch.StartNew();
            colorConfidenceMap = classifier.Classify(sourceMatResized, out var _, out var _, out var _);
            sw.Stop();
            Console.WriteLine($"Took {sw.ElapsedMilliseconds}ms");

            var color = colorConfidenceMap.MaxBy(x => x.Value);
            Console.WriteLine($"Color is {color.Key}");

            Cv2.ImShow(nameof(sourceMatResized), sourceMatResized);
            Cv2.ImShow(nameof(grabCut), grabCut);
            Cv2.ImShow(nameof(x), x);
            Cv2.ImShow(nameof(y), y);

            Cv2.WaitKey();
            Cv2.DestroyAllWindows();*/
        }
    }
}
