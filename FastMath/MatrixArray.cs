using ILGPU;
using ILGPU.Runtime;

namespace FastMath
{
    public class MatrixArray
    {
        public string Name { get; set; }
        public MemoryBuffer3D<float, Stride3D.DenseXY> Buffer { get; set; }
        public int Columns => (int)Buffer.Extent.X;
        public int Rows => (int)Buffer.Extent.Y;
        public int Length => (int)Buffer.Extent.Z;
        public float[,,] Current => Buffer.GetAsArray3D();
    }
}
