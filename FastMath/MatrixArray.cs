using ILGPU;
using ILGPU.Runtime;
using System;

namespace FastMath
{
    public class MatrixArray : Matrix
    {
        public MatrixArray(string name, MemoryBuffer3D<float, Stride3D.DenseXY> buffer) 
            : base(name, buffer)
        {
        }

        public float[,,] Current => Buffer.GetAsArray3D();

    }
}
