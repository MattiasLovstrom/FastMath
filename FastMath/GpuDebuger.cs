using System.Text;

namespace FastMath
{
    public class GpuDebugger : Gpu
    {
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
            var blockViewer = new BlockViewer();
            blockViewer.Add($"{result.Name} = {matrix.Name} * {value}  => ");
            blockViewer.Add(result);
            Console.Out.WriteLine(blockViewer.ToString());

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
            DisplayBinaryOperation("*" , matrix, matrix2, result);

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
            var blockViewer = new BlockViewer();
            blockViewer.Add($"{result.Name} = |{matrix.Name}|T =");
            blockViewer.Add(result);
            Console.Out.WriteLine(blockViewer.ToString());

            return result;
        }

        private static void DisplayMatrix(Matrix matrix)
        {
            var blockViewer = new BlockViewer();
            blockViewer.Add($"{matrix.Name} =");
            blockViewer.Add(matrix);
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


    public class BlockViewer
    {
        private List<BlockView> _blocks = new();

        public void Add(Matrix matrix)
        {
            _blocks.Add(new BlockView(matrix));
        }

        public void Add(string text)
        {
            _blocks.Add(new BlockView(text));
        }

        public override string ToString()
        {
            var result = new StringBuilder();
            var maxHeight = _blocks.Select(b => b.Rows.Count).Max();
            foreach (var block in _blocks)
            {
                block.PadRows(maxHeight);
            }

            for (var row = 0; row < maxHeight; row++)
            {
                foreach (var block in _blocks)
                {
                    result.Append(block.Rows.Count > row 
                        ? block.Rows[row] 
                        : "".PadRight(block.Width));
                }

                if (row < maxHeight-1)
                {
                    result.AppendLine();
                }
            }

            return result.ToString();
        }
    }

    public class BlockView
    {
        public readonly List<string> Rows = new();
        public readonly int Width;

        public BlockView (Matrix matrix)
        {
            var values = matrix.Current;
            for (var row = 0; row < matrix.Rows; row++)
            {
                var columns = new StringBuilder();
                columns.Append("|");
                for (int column = 0; column < matrix.Columns; column++)
                {
                    columns.Append(values[column, row].ToString(" #.00;-#.00; 0.00"));
                    columns.Append(" ");
                }
                columns.Append("|");
                Width = Math.Max(Width, columns.Length);
                Rows.Add(columns.ToString());
            }
        }

        public BlockView(string text)
        {
            Rows.Add(text);
            Width = text.Length;
        }

        public void PadRows(int maxHeight)
        {
            var linesToAdd = maxHeight - Rows.Count;
            for (var i = 0; i <  linesToAdd / 2; i++)
            {
                Rows.Insert(0, "".PadRight(Width));
            }

            for (var i = Rows.Count; i < linesToAdd; i++)
            {
                Rows.Add("".PadRight(Width));
            }
        }
    }
}
