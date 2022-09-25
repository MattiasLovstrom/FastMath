using ILGPU;
using ILGPU.Runtime;

namespace FastMath
{
    public class Matrix 
    {
        public Matrix(string name, MemoryBuffer3D<float, Stride3D.DenseXY> buffer)
        {
            Name = name;
            Buffer = buffer;
        }

        public string Name { get; set; }
        public MemoryBuffer3D<float, Stride3D.DenseXY> Buffer { get; set; }
        public int Columns => (int)Buffer.Extent.X;
        public int Rows => (int)Buffer.Extent.Y;
        public int Length => (int)Buffer.Extent.Z;

        public float[] SerializedData()
        {
            var result = new float[Length * Rows * Columns];
            var data3d = Buffer.GetAsArray3D();
            for (var index = 0; index < Length; index++)
            {
                for (var row = 0; row < Rows; row++)
                {
                    for (var column = 0; column < Columns; column++)
                    {
                        result[index * Rows * Columns + row * Columns + column] = data3d[column, row, index];
                    }
                }
            }

            return result;
        }

        public new float[,] Current
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

        public override string ToString()
        {
            var blockViewer = new BlockViewer();
            blockViewer.Add($"{Name} = ");
            blockViewer.Add(this);
            return blockViewer.ToString();
        }
    }
}