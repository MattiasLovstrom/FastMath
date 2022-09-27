namespace FastMath
{
    public class GpuDebugger : Gpu
    {
        public const string FloatFormat = " #.00;-#.00; 0.00";

        public override Matrix Fill(Matrix matrix, params float[] serializedData)
        {
            base.Fill(matrix, serializedData);
            DisplayMatrix(matrix);

            return matrix;
        }

        public override Matrix Fill(Matrix matrix, Func<IEnumerable<float>> serializedDataFunction)
        {
            base.Fill(matrix, serializedDataFunction);
            DisplayMatrix(matrix);

            return matrix;
        }

        public override async Task<Matrix> MulAsync(Matrix matrix, float value, Matrix result)
        {
            await base.MulAsync(matrix, value, result);
            DisplayBinaryOperation("*", matrix, value, result);

            return result;
        }

        public override async Task<Matrix> MulAsync(Matrix matrix, Matrix matrix2, Matrix result)
        {
            try
            {
                await base.MulAsync(matrix, matrix2, result);
            }
            catch (Exception)
            {
                DisplayBinaryOperation("*", matrix, matrix2, result);
                throw;
            }
            DisplayBinaryOperation("*", matrix, matrix2, result);

            return result;
        }
        public override async Task<Matrix> SubAsync(Matrix matrix1, Matrix matrix2, Matrix result)
        {
            await base.SubAsync(matrix1, matrix2, result);
            DisplayBinaryOperation("-", matrix1, matrix2, result);

            return result;
        }

        public override async Task<Matrix> TransposeAsync(Matrix matrix, Matrix result)
        {
            await base.TransposeAsync(matrix, result);
            for (var index = 0; index < matrix.Length; index++)
            {
                var blockViewer = new BlockViewer();
                if (matrix.Length == 0)
                {
                    blockViewer.Add($"{result.Name} = |{matrix.Name}|T = ");
                }
                else
                {
                    blockViewer.Add($"{result.Name}[{index}] = |{matrix.Name}[{index}]|T = ");
                }
                blockViewer.Add(result);
                Console.Out.WriteLine(blockViewer.ToString());
            }

            return result;
        }

        private static void DisplayMatrix(Matrix matrix)
        {
            for (var index = 0; index < matrix.Length; index++)
            {
                var blockViewer = new BlockViewer();
                if (matrix.Length == 0)
                {
                    blockViewer.Add($"{matrix.Name} = ");
                }
                else
                {
                    blockViewer.Add($"{matrix.Name}[{index}] = ");
                }
                blockViewer.Add(matrix);
                Console.Out.WriteLine(blockViewer.ToString());
            }
        }

        private static void DisplayBinaryOperation(string operatorString, Matrix matrix1, float value, Matrix result)
        {
            var blockViewer = new BlockViewer();
            blockViewer.Add($"{result.Name} = {matrix1.Name} {operatorString} {value}  => ");
            blockViewer.Add(result);
            blockViewer.Add(" = ");
            blockViewer.Add(matrix1);
            blockViewer.Add($" {operatorString} ");
            blockViewer.Add(value.ToString(FloatFormat));
            Console.Out.WriteLine(blockViewer.ToString());
        }

        private static void DisplayBinaryOperation(string operatorString, Matrix matrix1, Matrix matrix2, Matrix result)
        {
            var blockViewer = new BlockViewer();
            blockViewer.Add($"{result.Name} = {matrix1.Name} {operatorString} {matrix2.Name}  => ");
            blockViewer.Add(result);
            blockViewer.Add(" = ");
            blockViewer.Add(matrix1);
            blockViewer.Add($" {operatorString} ");
            blockViewer.Add(matrix2);
            Console.Out.WriteLine(blockViewer.ToString());
        }
    }
}
