using OpenCvSharp;

namespace ObjectColorClassifier
{
    internal class Program
    {
        static void Main(string[] args)
        {
            const ulong seed = 420;
            Cv2.SetTheRNG(seed);

            // Load the image
            using var sourceMat = Cv2.ImRead(@"C:\Users\ibenovic\Desktop\person_cropped_10.jpg");
            const string modelPath = @"C:\Users\ibenovic\Desktop\rgb_color_classifier.onnx";

            using var classifier = new ColorClassifierLib.ObjectColorClassifier(modelPath);
            var colorConfidenceMap = classifier.Classify(sourceMat, out var grabCut, out var x, out var y);

            Cv2.ImShow(nameof(sourceMat), sourceMat);
            Cv2.ImShow(nameof(grabCut), grabCut);
            Cv2.ImShow(nameof(x), x);
            Cv2.ImShow(nameof(y), y);

            var color = colorConfidenceMap.MaxBy(x => x.Value);

            Console.WriteLine($"Color is {color.Key}");

            Cv2.WaitKey();
            Cv2.DestroyAllWindows();
        }


    }
}
