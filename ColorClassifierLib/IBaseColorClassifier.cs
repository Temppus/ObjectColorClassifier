namespace ColorClassifierLib
{
    public interface IBaseColorClassifier
    {
        public IDictionary<BaseColor, float> GetBaseColorConfidences(byte r, byte g, byte b);
    }
}
