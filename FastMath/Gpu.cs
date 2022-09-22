using ILGPU;
using ILGPU.Runtime;

namespace FastMath
{
    public class Gpu : IProcessor, IDisposable
    {
        private Dictionary<string, Matrix> _matrixes = new();
        private static readonly Context Context;
        private static readonly Accelerator Accelerator;

        static Gpu()
        {
#if DEBUGCPU
            Context = Context.CreateDefault();
            Accelerator = Context.GetPreferredDevice(true).CreateAccelerator(Context);
#else
            Context = Context.Create(b => b.Default());
            Accelerator = Context.GetPreferredDevice(false).CreateAccelerator(Context);
#endif
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

            MatrixSubKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixSubAcceleratedKernel);

            MatrixMulKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                float,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixMulAcceleratedKernel);

            MatrixPow2Kernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixPow2AcceleratedKernel);

            MatrixTransposeKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixTransposeAcceleratedKernel);
        }

        public virtual Matrix GetOrCreate(string name, int columns, int rows)
        {
            if (_matrixes.ContainsKey(name)) return _matrixes[name];

            var matrix = new Matrix
            {
                Name = name,
                Buffer = Accelerator.Allocate2DDenseX<float>(new Index2D(columns, rows))
            };
            _matrixes.Add(name, matrix);

            return _matrixes[name];
        }

        public Matrix Get(string name)
        {
            if (_matrixes.ContainsKey(name)) return _matrixes[name];

            return null;
        }

        public virtual Matrix Fill(Matrix matrix, params float[] serializedData)
        {
            var fillPointer = 0;
            var data = new float[matrix.Columns, matrix.Rows];
            for (int row = 0; row < matrix.Rows; row++)
            {
                for (int column = 0; column < matrix.Columns; column++)
                {
                    data[column, row] = serializedData[fillPointer++];
                    if (fillPointer >= serializedData.Length) fillPointer = 0;
                }
            }
            matrix.Buffer.CopyFromCPU(data);

            return matrix;
        }

        public virtual Matrix Fill(Matrix matrix, Func<IEnumerable<float>> serializedDataFunction)
        {
            var serializedData = serializedDataFunction.Invoke();

            var t = serializedData.GetEnumerator();
            var data = new float[matrix.Columns, matrix.Rows];
            for (int row = 0; row < matrix.Rows; row++)
            {
                for (int column = 0; column < matrix.Columns; column++)
                {
                    data[column, row] = t.Current;
                    if (!t.MoveNext()) t.Reset();
                }
            }
            matrix.Buffer.CopyFromCPU(data);

            return matrix;
        }

        public Task<Matrix> SubAsync(Matrix matrix1, Matrix matrix2, string result)
        {
            throw new NotImplementedException();
        }

        public virtual async Task<Matrix> MulAsync(
            Matrix matrix,
            float value,
            Matrix result)
        {
            await Task.Run(() => MatrixMulKernel(
                result.Buffer.Extent.ToIntIndex(),
                matrix.Buffer.View,
                value,
                result.Buffer.View));

            return result;
        }

        private static Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, float , ArrayView2D<float, Stride2D.DenseX>> MatrixMulKernel;
        public static void MatrixMulAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> matrix,
            float value,
            ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index] = matrix[index] * value;
        }

        public virtual async Task<Matrix> MulAsync(
            Matrix matrix,
            Matrix matrix2,
            Matrix result)
        {
            await Task.Run(()=> MatrixMultiplyKernel(
                result.Buffer.Extent.ToIntIndex(),
                matrix.Buffer.View,
                matrix2.Buffer.View,
                result.Buffer.View));

            return result;
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
                sum += matrix1[new Index2D(i, y)] * matrix2[new Index2D(x, i)];
                //Interop.WriteLine("m1 {0} {1} m2 {2} {3} = {4}", x,i,i,y,sum);
            }

            result[index] = sum;
        }

        public virtual async Task<Matrix> AddAsync(
            Matrix matrix1, 
            Matrix matrix2, 
            Matrix result)
        {
            await Task.Run(()=>MatrixAddKernel(
                result.Buffer.Extent.ToIntIndex(),
                matrix1.Buffer.View,
                matrix2.Buffer.View,
                result.Buffer.View));

            return result;
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

        public virtual async Task<Matrix> SubAsync(
            Matrix matrix1,
            Matrix matrix2,
            Matrix result)
        {
            await Task.Run(() => MatrixSubKernel(
                result.Buffer.Extent.ToIntIndex(),
                matrix1.Buffer.View,
                matrix2.Buffer.View,
                result.Buffer.View));

            return result;
        }

        private static Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> MatrixSubKernel;
        public static void MatrixSubAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> matrix1,
            ArrayView2D<float, Stride2D.DenseX> matrix2,
            ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index] = matrix1[index] - matrix2[index];
        }

        public virtual async Task<Matrix> Pow2Async(
            Matrix matrix,
            Matrix result)
        {
            await Task.Run(() => MatrixPow2Kernel(
                result.Buffer.Extent.ToIntIndex(),
                matrix.Buffer.View,
                result.Buffer.View));

            return result;
        }

        private static Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> MatrixPow2Kernel;
        public static void MatrixPow2AcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> matrix,
            ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index] = matrix[index] * matrix[index];
        }


        public virtual async Task<Matrix> TransposeAsync(
            Matrix matrix,
            Matrix result)
        {
            await Task.Run(() => MatrixTransposeKernel(
                matrix.Buffer.Extent.ToIntIndex(),
                matrix.Buffer.View,
                result.Buffer.View));

            return result;
        }

        private static Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> MatrixTransposeKernel;
        public static void MatrixTransposeAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> matrix,
            ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[new Index2D(index.Y, index.X)] = matrix[index];
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
