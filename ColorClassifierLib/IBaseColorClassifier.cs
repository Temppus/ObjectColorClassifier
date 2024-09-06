namespace ColorClassifierLib
{
    public interface IBaseColorClassifier
    {
        public BaseColor GetBaseColor(byte r, byte g, byte b);
    }
}
