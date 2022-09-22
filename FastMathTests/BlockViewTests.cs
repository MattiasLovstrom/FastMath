// ReSharper disable once CheckNamespace
namespace FastMath.Tests
{
    [TestClass]
    public class BlockViewTests
    {
        [TestMethod]
        public void BlockViewTest()
        {
            var testObject = new BlockView("test");
            Assert.AreEqual(1, testObject.Rows.Count);
        }
    }
}