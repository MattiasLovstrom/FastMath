namespace FastMath;

public interface IProcessor
{
    Matrix GetOrCreate(string name, int columns, int rows);
    Task<Matrix> MulAsync(Matrix matrix, Matrix matrix2, Matrix result);
    Task<Matrix> MulAsync(Matrix matrix, float value, Matrix result);
    MatrixArray? Get(string name);                              
    Task<Matrix> AddAsync(Matrix matrix1, Matrix matrix2, Matrix result);
    Task<Matrix> SubAsync(Matrix matrix1, Matrix matrix2, Matrix result);
    Task<Matrix> Pow2Async(Matrix matrix, Matrix result);
    Matrix Fill(Matrix matrix, params float[] serializedData);
    Matrix Fill(Matrix matrix, Func<IEnumerable<float>> serializedDataFunction);
    Task<Matrix> TransposeAsync(Matrix matrix, Matrix result);
}