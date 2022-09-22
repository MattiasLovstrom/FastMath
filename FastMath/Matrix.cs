using ILGPU;
using ILGPU.Runtime;

namespace FastMath
{
    public class Matrix
    {
        public Matrix Fill(params float[] serializedData)
        {
            var fillPointer = 0;
            var data = new float[Columns, Rows];
            for (var row = 0; row < Rows; row++)
            {
                for (var column = 0; column < Columns; column++)
                {
                    data[column, row] = serializedData[fillPointer++];
                    if (fillPointer >= serializedData.Length) fillPointer = 0;
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
        public string Name { get; set; }

        public float[] SerializedData()
        {
            var result = new float[Rows * Columns];
            var data = Current;
            for (var row = 0; row < Rows; row++)
            {
                for (var column = 0; column < Columns; column++)
                {
                    result[row*Columns+ column] = data[column, row];
                }
            }

            return result;
        }

        public override string ToString()
        {
            var blockViewer = new BlockViewer();
            blockViewer.Add($"{Name} = ");
            blockViewer.Add(this);
            return blockViewer.ToString();
        }
    }
}