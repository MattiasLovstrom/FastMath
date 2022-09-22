// ReSharper disable once CheckNamespace
namespace FastMath.Tests
{
    [TestClass]
    public class BlockViewerTests
    {
        [TestMethod]
        public void AddTest()
        {
            var testObject = new BlockViewer();
            testObject.Add("test");

            Assert.AreEqual("test", testObject.ToString());
        }
    }
}