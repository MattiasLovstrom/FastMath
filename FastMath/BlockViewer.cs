using System.Text;

namespace FastMath;

public class BlockViewer
{
    private readonly List<BlockView> _blocks = new();

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

            if (row < maxHeight - 1)
            {
                result.AppendLine();
            }
        }

        return result.ToString();
    }
}