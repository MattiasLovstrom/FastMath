namespace MlMath.Tests
{
    [TestClass()]
    public class MatrixTests
    {
        [TestMethod]
        public void MatrixTest()
        {
            using var gpu = new Gpu(); 
            var testObject = gpu.GetOrCreate("t", 3, 4).Fill(
                00, 01, 02,
                10, 11, 12,
                20, 21, 22, 
                30, 31, 32);
            Assert.AreEqual(00, testObject.Current[0, 0]);
            Assert.AreEqual(32, testObject.Current[2, 3]);
        }

        [TestMethod]
        public async Task MulSimpleTest()
        {
            using var gpu = new Gpu();
            var a = gpu.GetOrCreate("a", 1, 1);
            a.Fill(
                1);
            var b = gpu.GetOrCreate("b", 1, 1);
            b.Fill(
                2);

            var result = gpu.GetOrCreate("result", 1, 1);

            await gpu.MulAsync(a, b, result);
            var r = result.Current;

            Assert.AreEqual(1 * 2, r[0, 0]);
        }

        [TestMethod]
        public async Task MulTest()
        {
            using var gpu = new Gpu();
            var a = gpu.GetOrCreate("a", 1, 2);
            a.Fill(
                1,
                2);
            var b = gpu.GetOrCreate("b", 2, 1);
            b.Fill(
                3,4);
            
            var result = gpu.GetOrCreate("result", 2, 2);

            var t =  gpu.MulAsync(a, b, result);
            Task.WaitAll(t);
            var r = result.Current;

            Assert.AreEqual(1 * 3, r[0, 0]);
            Assert.AreEqual(1 * 4, r[0, 1]);
            Assert.AreEqual(2 * 3, r[1, 0]);
            Assert.AreEqual(2 * 4, r[1, 1]);
        }
    }
}