
/**
 * Helper function to select stock data
 * Returns an array of values
 * @param {array} rows
 * @param {integer} index
 * index 0 - Date
 * index 1 - Open
 * index 2 - High
 * index 3 - Low
 * index 4 - Close
 * index 5 - Volume
 */
console.log("Hello init test");
// Submit Button handler
function handleSubmit() {
  console.log("in handle submit");
  // @TODO: YOUR CODE HERE
  // Prevent the page from refreshing
  //d3.event.preventDefault();

  // Select the input value from the form
  //var inval = d3.select("#stockInput").on("change", updatePlotly);
  var inval = d3.select("#selDatasetCntKc").node().value;
  console.log(inval);
  buildPlot(inval);
  // clear the input value

  // Build the plot with the new stock
}
handleSubmit();
d3.select("#selDatasetCntKc").on("select", handleSubmit);
function buildPlot(stock) {

  d3.csv("/static/data/"+stock+".csv").then(function(tvData) {
    // Grab values from the response json object to build the plots

    //tvData.forEach(function(data) {
     // data.Date = +data.Date;
      //data.Open = +data.Open;
    //});



    // Print the names of the columns
  //  console.log(data.dataset.column_names);
    // Print the data for each day
   // console.log(data.dataset.data);
    // Use map() to build an array of the the dates
    // var dates =
    // Use map() to build an array of the closing prices
    // var closingPrices =
    var dates = tvData.map(data => data.Date);
    console.log(dates);
    var openPrices = tvData.map(data => data.Open);
    console.log(openPrices);
    
 
    var startDate = "2016-01-19";
    var endDate = "2021-01-15";
    var trace1 = {
      type: "scatter",
      mode: "lines",
      name: stock,
      x: dates,
      y: openPrices,
      line: {
        color: "#17BECF"
      }
    };

    var data = [trace1];
    console.log(trace1);
    // Print the data for each day
    console.log(stock);
    var layout = {
      title: `${stock} closing prices`,
      xaxis: {
        range: [startDate, endDate],
        type: "date"
      },
      yaxis: {
        autorange: true,
        type: "linear"
      }
    };

    Plotly.newPlot("plot", data, layout);

  });
}

// Add event listener for submit button
// @TODO: YOUR CODE HERE
