using System.Diagnostics;
using FastMath;

namespace FastMathTests
{
    [TestClass]
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
            gpu.Fill(a, 1);
            var b = gpu.GetOrCreate("b", 1, 1);
            gpu.Fill(b, 2);

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
                3, 4);

            var result = gpu.GetOrCreate("result", 2, 2);

            await gpu.MulAsync(a, b, result);
            var r = result.Current;
            //                       x  y
            Assert.AreEqual(1 * 3, r[0, 0]);
            Assert.AreEqual(1 * 4, r[1, 0]);
            Assert.AreEqual(2 * 3, r[0, 1]);
            Assert.AreEqual(2 * 4, r[1, 1]);
        }


        [TestMethod]
        public async Task MulTest2()
        {
            using var gpu = new Gpu();
            var a = gpu.GetOrCreate("a", 3, 2);
            a.Fill(
                11, 21, 31,
                12, 22, 32);
            var b = gpu.GetOrCreate("b", 2, 3);
            b.Fill(
                11, 21,
                12, 22,
                13, 23);

            var result = gpu.GetOrCreate("result", 2, 2);

            await gpu.MulAsync(a, b, result);
            var r = result.Current;
            Assert.AreEqual(11 * 11 + 21 * 12 + 31 * 13, r[0, 0]);
            Assert.AreEqual(11 * 21 + 21 * 22 + 31 * 23, r[1, 0]);
            Assert.AreEqual(12 * 11 + 22 * 12 + 32 * 13, r[0, 1]);
            Assert.AreEqual(12 * 21 + 22 * 22 + 32 * 23, r[1, 1]);
        }

        [TestMethod]
        public async Task MulLargeTest()
        {
            var sw = new Stopwatch();
            sw.Start();

            using var gpu = new Gpu();
            var a = gpu.GetOrCreate("a", 1000, 1000);
            a.Fill(1);
            var b = gpu.GetOrCreate("b", 1000, 1000);
            b.Fill(2);
            var result = gpu.GetOrCreate("result", 1000, 1000);
            Console.Out.WriteLine($"Alloc: {sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var t = gpu.MulAsync(a, b, result);
            Console.Out.WriteLine($"Mul: {sw.ElapsedMilliseconds}ms");
            sw.Restart();
            var t1 = gpu.MulAsync(a, b, result);
            Console.Out.WriteLine($"Mul: {sw.ElapsedMilliseconds}ms");
            sw.Restart();
            Task.WaitAll(t, t1);
            var r = result.Current;
            sw.Restart();
            await Console.Out.WriteLineAsync($"Result: {sw.ElapsedMilliseconds}ms");

            Assert.AreEqual(2000, r[0, 0]);
            Assert.AreEqual(2000, r[0, 1]);
            Assert.AreEqual(2000, r[1, 0]);
            Assert.AreEqual(2000, r[999, 999]);
        }

        [TestMethod]
        public async Task AsyncTest()
        {
            using var gpu = new Gpu();
            var a = gpu.GetOrCreate("a", 1000, 1000);
            a.Fill(1);
            var b = gpu.GetOrCreate("b", 1000, 1000);
            b.Fill(2);
            var t1 = Task.Run(async () =>
            {
                for (var i = 0; i < 1000; i++)
                {
                    await gpu.AddAsync(a, b, a);
                }
            });
            Assert.AreNotEqual(2001, a.Current[0, 0]);
            await t1;

            var r = a.Current;
            Assert.AreEqual(2001, r[0, 0]);
            Assert.AreEqual(2001, r[0, 1]);
            Assert.AreEqual(2001, r[1, 0]);
            Assert.AreEqual(2001, r[999, 999]);
        }

        [TestMethod]
        public async Task VectorMulMatrixTest()
        {
            using var gpu = new Gpu();
            var a = gpu.GetOrCreate("a", 10, 1);
            gpu.Fill(a, FillMe);
            var b = gpu.GetOrCreate("b", 9, 10);
            gpu.Fill(b, FillMe);
            var result = gpu.GetOrCreate("result", 9, 1);

            await gpu.MulAsync(a, b, result);
        }

        private IEnumerable<float> FillMe()
        {
            var i = 0;
            while (true)
            {
                yield return ++i;
            }
        }

        [AssemblyCleanup]
        public static void AssemblyCleanup()
        {
            Gpu.Close();
        }
    }
}