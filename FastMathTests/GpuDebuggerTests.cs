using Microsoft.VisualStudio.TestTools.UnitTesting;
using FastMath;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FastMath.Tests
{
    [TestClass()]
    public class GpuDebuggerTests
    {
        [TestMethod()]
        public async Task TransposeAsyncTest()
        {
            var debugger = new GpuDebugger();
            var calculator = new Calculator(debugger);
            var a = debugger.GetOrCreate("2_2_2", 2, 2, 2);
            calculator.Fill(a, 1, 2, 3, 4);
            await calculator.TransposeAsync(a);
        }
    }
}