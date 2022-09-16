using ILGPU;
using ILGPU.Runtime;

namespace MlMath
{
    public class Matrix
    {
        public Matrix Fill(params float[] serializedData)
        {
            var data = new float[Columns, Rows];
            for (int row = 0; row < Rows; row++)
            {
                for (int column = 0; column < Columns; column++)
                {
                    data[column, row] = serializedData[Columns * row + column];
                }
            }
            Buffer.CopyFromCPU(data);

            return this;
        }

        public int Columns => (int)Buffer.Extent.X;

        public int Rows => (int)Buffer.Extent.Y;


        //public float[,] Values { get; set; }
        public MemoryBuffer2D<float, Stride2D.DenseX> Buffer { get; set; }

        public float[,] Current => Buffer.GetAsArray2D();
    }
}