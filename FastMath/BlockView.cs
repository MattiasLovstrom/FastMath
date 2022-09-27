using System.Text;

namespace FastMath;

public class BlockView
{
    public readonly List<string> Rows = new();
    public readonly int Width;

    public BlockView(Matrix matrix)
    {
        var values = matrix.Current;
        var maxLength = 0;
        for (var row = 0; row < matrix.Rows; row++)
        {
            for (int column = 0; column < matrix.Columns; column++)
            {
                maxLength = Math.Max(maxLength, values[column, row].ToString(GpuDebugger.FloatFormat).Length);
            }
        }

        var padding = "".PadLeft(maxLength - " #.00".Length);

        for (var row = 0; row < matrix.Rows; row++)
        {
            var columns = new StringBuilder();
            columns.Append("|");
            for (int column = 0; column < matrix.Columns; column++)
            {
                var value = values[column, row].ToString(" #.00;-#.00; 0.00");
                columns.Append("".PadRight(maxLength - value.Length));
                columns.Append(value);
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
        for (var i = 0; i < linesToAdd / 2; i++)
        {
            Rows.Insert(0, "".PadRight(Width));
        }

        for (var i = Rows.Count; i < linesToAdd; i++)
        {
            Rows.Add("".PadRight(Width));
        }
    }
}