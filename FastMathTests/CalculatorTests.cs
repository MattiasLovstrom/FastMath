using Microsoft.VisualStudio.TestTools.UnitTesting;
using FastMath;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FastMath.Tests
{
    [TestClass]
    public class CalculatorTests
    {
        [TestMethod]
        public async Task AddAsyncTest()
        {
            using var gpu = new GpuDebugger();
            var calculator = new Calculator(gpu);
            var a = gpu.GetOrCreate("a", 1, 1);
            gpu.Fill(a, 1);
            var b = gpu.GetOrCreate("b", 1, 1);
            gpu.Fill(b, 2);

            var result = await calculator.AddAsync(a, b);
            Console.Out.WriteLine(result);
            var r = result.Current;

            Assert.AreEqual(1 + 2, r[0, 0]);
        }
    }
}