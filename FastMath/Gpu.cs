using ILGPU;
using ILGPU.Runtime;

namespace FastMath
{
    public class Gpu : IProcessor, IDisposable
    {
        private Dictionary<string, MatrixArray> _matrixes = new();
        //private Dictionary<string, MatrixArray> _arrays = new();
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
                Index3D,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>>(
                MatrixMultiplyAcceleratedKernel);

            MatrixMultiplyArrayKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index3D,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>>(
                MatrixMultiplyArrayAcceleratedKernel);

            MatrixAddKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index3D,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>>(
                MatrixAddAcceleratedKernel);

            MatrixSubKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index3D,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>>(
                MatrixSubAcceleratedKernel);

            MatrixMulKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index3D,
                ArrayView3D<float, Stride3D.DenseXY>,
                float,
                ArrayView3D<float, Stride3D.DenseXY>>(
                MatrixMulAcceleratedKernel);

            MatrixMulArrayKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index3D,
                ArrayView3D<float, Stride3D.DenseXY>,
                float,
                ArrayView3D<float, Stride3D.DenseXY>>(
                MatrixMulArrayAcceleratedKernel);

            MatrixPow2Kernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index3D,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>>(
                MatrixPow2AcceleratedKernel);

            MatrixTransposeKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index3D,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>>(
                MatrixTransposeAcceleratedKernel);
        }

        public virtual Matrix GetOrCreate(string name, int columns, int rows)
        {
            return GetOrCreate(name, columns, rows, 1);
        }

        public virtual MatrixArray GetOrCreate(string name, int columns, int rows, int length)
        {
            if (_matrixes.ContainsKey(name)) return _matrixes[name];

            var matrix = new MatrixArray(
                name,
                Accelerator.Allocate3DDenseXY<float>(new Index3D(columns, rows, length)));
            _matrixes.Add(name, matrix);

            return _matrixes[name];
        }

        public MatrixArray? Get(string name)
        {
            if (_matrixes.ContainsKey(name)) return _matrixes[name];

            return null;
        }

        public virtual Matrix Fill(Matrix matrix, params float[] serializedData)
        {
            var fillPointer = 0;
            var data = new float[matrix.Columns, matrix.Rows, matrix.Length];

            for (var index = 0; index < matrix.Length; index++)
            {
                for (var row = 0; row < matrix.Rows; row++)
                {
                    for (var column = 0; column < matrix.Columns; column++)
                    {
                        data[column, row, index] = serializedData[fillPointer++];
                        if (fillPointer >= serializedData.Length) fillPointer = 0;
                    }
                }
            }
            matrix.Buffer.CopyFromCPU(data);

            return matrix;
        }

        public virtual Matrix Fill(Matrix matrix, Func<IEnumerable<float>> serializedDataFunction)
        {
            var serializedData = serializedDataFunction.Invoke();

            var t = serializedData.GetEnumerator();
            var data = new float[matrix.Columns, matrix.Rows, 1];
            for (int row = 0; row < matrix.Rows; row++)
            {
                for (int column = 0; column < matrix.Columns; column++)
                {
                    data[column, row, 0] = t.Current;
                    if (!t.MoveNext()) t.Reset();
                }
            }
            matrix.Buffer.CopyFromCPU(data);

            return matrix;
        }

        public virtual MatrixArray Fill(MatrixArray array, params float[] serializedData)
        {
            var fillPointer = 0;
            var data = new float[array.Columns, array.Rows, array.Length];
            for (var index = 0; index < array.Length; index++)
            {
                for (int row = 0; row < array.Rows; row++)
                {
                    for (int column = 0; column < array.Columns; column++)
                    {
                        data[column, row, index] = serializedData[fillPointer++];
                        if (fillPointer >= serializedData.Length) fillPointer = 0;
                    }
                }
            }
            array.Buffer.CopyFromCPU(data);

            return array;
        }

        #region Sub
        public Task<Matrix> SubAsync(Matrix matrix1, Matrix matrix2, string result)
        {
            throw new NotImplementedException();
        }
        #endregion

        #region Mul
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

        private static Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, float, ArrayView3D<float, Stride3D.DenseXY>> MatrixMulKernel;
        public static void MatrixMulAcceleratedKernel(
            Index3D index,
            ArrayView3D<float, Stride3D.DenseXY> matrix,
            float value,
            ArrayView3D<float, Stride3D.DenseXY> result)
        {
            result[index] = matrix[index] * value;
        }
        public virtual async Task<MatrixArray> MulAsync(
            MatrixArray matrix,
            float value,
            MatrixArray result)
        {
            await Task.Run(() => MatrixMulArrayKernel(
                result.Buffer.Extent.ToIntIndex(),
                matrix.Buffer.View,
                value,
                result.Buffer.View));

            return result;
        }

        private static Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, float, ArrayView3D<float, Stride3D.DenseXY>> MatrixMulArrayKernel;
        public static void MatrixMulArrayAcceleratedKernel(
            Index3D index,
            ArrayView3D<float, Stride3D.DenseXY> matrixArray,
            float value,
            ArrayView3D<float, Stride3D.DenseXY> resultArray)
        {
            resultArray[index] = matrixArray[index] * value;
        }
        public virtual async Task<Matrix> MulAsync(
            Matrix matrix,
            Matrix matrix2,
            Matrix result)
        {
            await Task.Run(() => MatrixMultiplyKernel(
                result.Buffer.Extent.ToIntIndex(),
                matrix.Buffer.View,
                matrix2.Buffer.View,
                result.Buffer.View));

            return result;
        }

        private static Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> MatrixMultiplyKernel;
        public static void MatrixMultiplyAcceleratedKernel(
            Index3D index,
            ArrayView3D<float, Stride3D.DenseXY> matrix1,
            ArrayView3D<float, Stride3D.DenseXY> matrix2,
            ArrayView3D<float, Stride3D.DenseXY> result)
        {
            var x = index.X;
            var y = index.Y;
            var sum = 0.0f;
            for (var i = 0; i < matrix1.IntExtent.X; i++)
            {
                sum += matrix1[new Index3D(i, y, 0)] * matrix2[new Index3D(x, i, 0)];
            }

            result[index] = sum;
        }

        public virtual async Task<MatrixArray> MulAsync(
           MatrixArray matrixArray,
           Matrix matrix,
           MatrixArray resultArray)
        {
            await Task.Run(() => MatrixMultiplyArrayKernel(
                resultArray.Buffer.Extent.ToIntIndex(),
                matrixArray.Buffer.View,
                matrix.Buffer.View,
                resultArray.Buffer.View));

            return resultArray;
        }

        private static Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> MatrixMultiplyArrayKernel;
        public static void MatrixMultiplyArrayAcceleratedKernel(
            Index3D index,
            ArrayView3D<float, Stride3D.DenseXY> matrixArray,
            ArrayView3D<float, Stride3D.DenseXY> matrix,
            ArrayView3D<float, Stride3D.DenseXY> resultArray)
        {
            var x = index.X;
            var y = index.Y;
            var z = index.Z;
            var sum = 0.0f;
            for (var i = 0; i < matrixArray.IntExtent.X; i++)
            {
                sum += matrixArray[new Index3D(i, y, z)] * matrix[new Index3D(x, i, 0)];
            }

            resultArray[index] = sum;
        }
        #endregion

        #region Add
        public virtual async Task<Matrix> AddAsync(
            Matrix matrix1,
            Matrix matrix2,
            Matrix result)
        {
            await Task.Run(() => MatrixAddKernel(
                result.Buffer.Extent.ToIntIndex(),
                matrix1.Buffer.View,
                matrix2.Buffer.View,
                result.Buffer.View));

            return result;
        }

        private static Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> MatrixAddKernel;
        public static void MatrixAddAcceleratedKernel(
            Index3D index,
            ArrayView3D<float, Stride3D.DenseXY> matrix1,
            ArrayView3D<float, Stride3D.DenseXY> matrix2,
            ArrayView3D<float, Stride3D.DenseXY> result)
        {
            result[index] = matrix1[index] + matrix2[index];
        }
        #endregion

        #region Sub
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

        private static Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> MatrixSubKernel;
        public static void MatrixSubAcceleratedKernel(
            Index3D index,
            ArrayView3D<float, Stride3D.DenseXY> matrix1,
            ArrayView3D<float, Stride3D.DenseXY> matrix2,
            ArrayView3D<float, Stride3D.DenseXY> result)
        {
            result[index] = matrix1[index] - matrix2[index];
        }
        #endregion

        #region Pow2
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

        private static Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> MatrixPow2Kernel;
        public static void MatrixPow2AcceleratedKernel(
            Index3D index,
            ArrayView3D<float, Stride3D.DenseXY> result,
            ArrayView3D<float, Stride3D.DenseXY> matrix)
        {
            result[index] = matrix[index] * matrix[index];
        }
        #endregion

        #region Transpose
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

        private static Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> MatrixTransposeKernel;
        public static void MatrixTransposeAcceleratedKernel(
            Index3D index,
            ArrayView3D<float, Stride3D.DenseXY> matrix,
            ArrayView3D<float, Stride3D.DenseXY> result)
        {
            result[new Index3D(index.Y, index.X, index.Z)] = matrix[index];
        }
        #endregion

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
