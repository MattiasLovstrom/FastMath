using System.Numerics;

namespace FastMath
{
    public class Calculator
    {
        private IProcessor _processor;
        private static Calculator? _current;

        public Calculator(IProcessor processor)
        {
            _processor = processor;
        }

        public MatrixArray? Get(string name)
        {
            return _processor.Get(name);
        }

        public Matrix GetOrCreate(string name, int columns, int rows)
        {
            return _processor.GetOrCreate(name, columns, rows);
        }

        public Matrix Fill(Matrix matrix, params float[] serializedData)
        {
            return _processor.Fill(matrix, serializedData);
        }

        public Matrix Fill(Matrix matrix, Func<IEnumerable<float>> serializedDataFunction)
        {
            return _processor.Fill(matrix, serializedDataFunction);
        }

        public async Task<Matrix> MulAsync(Matrix matrix1, Matrix matrix2)
        {
            var result = _processor.GetOrCreate($"{matrix1.Name}**{matrix2.Name}", matrix2.Columns, matrix1.Rows);

            return await MulAsync(matrix1, matrix2, result);
        }

        public async Task<Matrix> MulAsync(Matrix matrix1, Matrix matrix2, Matrix result)
        {
            if (matrix1.Columns != matrix2.Rows)
                throw new ArgumentException($"Matrix1.Columns must be equal with matrix2.Rows");
            if (matrix2.Columns != result.Columns)
                throw new ArgumentException($"Result.Columns must be same as matrix2.Columns");

            return await _processor.MulAsync(matrix1, matrix2, result);
        }

        public async Task<Matrix> MulAsync(Matrix matrix, float value)
        {
            var result = _processor.GetOrCreate($"{matrix.Name}*value", matrix.Columns, matrix.Rows);

            return await _processor.MulAsync(matrix, value, result);
        }

        public async Task<Matrix> MulAsync(Matrix matrix, float value, Matrix result)
        {
            return await _processor.MulAsync(matrix, value, result);
        }

        public async Task<Matrix> AddAsync(Matrix matrix1, Matrix matrix2)
        {
            var result = _processor.GetOrCreate($"{matrix1.Name}+{matrix2.Name}", matrix1.Columns, matrix1.Rows);

            return await AddAsync(matrix1, matrix2, result);
        }

        public async Task<Matrix> AddAsync(Matrix matrix1, Matrix matrix2, Matrix result)
        {
            return await _processor.AddAsync(matrix1, matrix2, result);
        }

        public async Task<Matrix> SubAsync(Matrix matrix1, Matrix matrix2)
        {
            var result = _processor.GetOrCreate($"{matrix1.Name}-{matrix2.Name}", matrix1.Columns, matrix1.Rows);
            
            return await SubAsync(matrix1, matrix2, result);
        }

        public async Task<Matrix> SubAsync(Matrix matrix1, Matrix matrix2, Matrix result)
        {
            return await _processor.SubAsync(matrix1, matrix2, result);
        }

        public async Task<Matrix> Pow2Async(Matrix matrix, Matrix result)
        {
            return await _processor.Pow2Async(matrix, result);
        }

        public async Task<Matrix> TransposeAsync(Matrix matrix)
        {
            var result = _processor.GetOrCreate($"{matrix.Name}T", matrix.Rows, matrix.Columns);
            
            return await TransposeAsync(matrix, result);
        }

        public async Task<Matrix> TransposeAsync(Matrix matrix, Matrix result)
        {
            if (matrix.Rows != result.Columns)
                throw new ArgumentException(
                    $"{nameof(matrix)} rows (here {matrix.Rows}) must be equal to {nameof(result)} columns (here {result.Columns})");
            if (matrix.Columns!= result.Rows)
                throw new ArgumentException(
                    $"{nameof(matrix)} columns (here {matrix.Columns}) must be equal to {nameof(result)} rows (here {result.Rows})");
            return await _processor.TransposeAsync(matrix, result);
        }
    }
}
