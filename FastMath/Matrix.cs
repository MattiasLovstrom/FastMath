using ILGPU;
using ILGPU.Runtime;

namespace FastMath
{
    public class Matrix
    {
        public int Columns => (int)Buffer.Extent.X;

        public int Rows => (int)Buffer.Extent.Y;

        public MemoryBuffer3D<float, Stride3D.DenseXY> Buffer { get; set; }

        public float[,] Current
        {
            get
            {
                var data3d = Buffer.GetAsArray3D(); 
                var result = new float[Columns, Rows];
                for(int i = 0; i < Columns; i++)
                {
                    for(int j = 0; j < Rows; j++)
                    {
                        result[i, j] = data3d[i, j, 0];
                    }
                }

                return result;
            }
        } 
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