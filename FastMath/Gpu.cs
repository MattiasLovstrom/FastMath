using ILGPU;
using ILGPU.Runtime;

namespace FastMath
{
    public class Gpu : IDisposable
    {
        private Dictionary<string, Matrix> _matrixes = new Dictionary<string, Matrix>();
        private static readonly Context Context;
        private static readonly Accelerator Accelerator;

        static Gpu()
        {
            Context = Context.Create(b => b.Default());
            Accelerator = Context.GetPreferredDevice(false).CreateAccelerator(Context);
            //Context = Context.CreateDefault();
            //Accelerator = Context.GetPreferredDevice(true).CreateAccelerator(Context);

            MatrixMultiplyKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixMultiplyAcceleratedKernel);

            MatrixAddKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixAddAcceleratedKernel);
        }

        public Matrix GetOrCreate(string name, int columns, int rows)
        {
            if (_matrixes.ContainsKey(name)) return _matrixes[name];

            var matrix = new Matrix()
            {
                Buffer = Accelerator.Allocate2DDenseX<float>(new Index2D(columns, rows))
            };
            _matrixes.Add(name, matrix);

            return _matrixes[name];
        }

        public async Task MulAsync(
            Matrix matrix1,
            Matrix matrix2,
            Matrix result)
        {
            await Task.Run(()=> MatrixMultiplyKernel(
                result.Buffer.Extent.ToIntIndex(),
                matrix1.Buffer.View,
                matrix2.Buffer.View,
                result.Buffer.View));
        }

        private static Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> MatrixMultiplyKernel;
        public static void MatrixMultiplyAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> matrix1,
            ArrayView2D<float, Stride2D.DenseX> matrix2,
            ArrayView2D<float, Stride2D.DenseX> result)
        {
            var x = index.X;
            var y = index.Y;
            var sum = 0.0f;
            for (var i = 0; i < matrix1.IntExtent.X; i++)
            {
                sum += matrix1[new Index2D(i, x)] * matrix2[new Index2D(y, i)];
                //Interop.WriteLine("m1 {0} {1} m2 {2} {3} = {4}", x,i,i,y,sum);
            }

            result[index] = sum;
        }

        public async Task AddAsync(
            Matrix matrix1,
            Matrix matrix2,
            Matrix result)
        {
            await Task.Run(()=>MatrixAddKernel(
                result.Buffer.Extent.ToIntIndex(),
                matrix1.Buffer.View,
                matrix2.Buffer.View,
                result.Buffer.View));
        }

        private static Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> MatrixAddKernel;
        public static void MatrixAddAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> matrix1,
            ArrayView2D<float, Stride2D.DenseX> matrix2,
            ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index] = matrix1[index] + matrix2[index];
        }

        public void Dispose()
        {
            foreach (var matrix in _matrixes)
            {
                matrix.Value.Buffer.Dispose();
            }

            _matrixes = null;
        }

        public static void Close()
        {
            Context.Dispose();
            Accelerator.Dispose();
        }
    }
}
