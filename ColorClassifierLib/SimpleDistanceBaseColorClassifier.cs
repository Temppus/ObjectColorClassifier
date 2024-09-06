namespace ColorClassifierLib
{
    public class SimpleDistanceBaseColorClassifier : IBaseColorClassifier
    {
        public BaseColor GetBaseColor(byte r, byte g, byte b)
        {
            var baseColors = new Dictionary<BaseColor, int[]>
            {
                { BaseColor.Red, new int[] { 220, 20, 60 } },
                { BaseColor.Green, new int[] { 0, 128, 0 } },
                { BaseColor.Blue, new int[] { 15, 82, 186 } },

                { BaseColor.Yellow, new int[] { 255, 255, 0 } },
                { BaseColor.Orange, new int[] { 255, 140, 0 } },
                { BaseColor.Pink, new int[] { 255, 105, 180 } },
                { BaseColor.Purple, new int[] { 153, 50, 204 } },
                { BaseColor.Brown, new int[] { 160, 82, 45 } },
                { BaseColor.Grey, new int[] { 105, 105, 105 } },

                { BaseColor.Black, new int[] { 0, 0, 0 } },
                { BaseColor.White, new int[] { 248, 248, 255 } }
            };

            int[] inputColor = { r, g, b };

            var closestColor = baseColors.MinBy(c => ColorDistance(inputColor, c.Value));

            return closestColor.Key;
        }

        // Method to calculate Euclidean distance between two RGB colors
        private static double ColorDistance(int[] color1, int[] color2)
        {
            return Math.Sqrt(Math.Pow(color1[0] - color2[0], 2) +
                             Math.Pow(color1[1] - color2[1], 2) +
                             Math.Pow(color1[2] - color2[2], 2));
        }
    }
}
