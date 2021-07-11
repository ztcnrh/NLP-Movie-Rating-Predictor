// Class selected genres & apply styling
d3.selectAll(".btn-outline-light").on("click", function() {
    var selection = d3.select(this);
    if (selection.classed("selected")) {
      selection.classed("selected", false);
    }
    else {
      selection.classed("selected",true);
    };
});


// Submission function
d3.select("#submit-btn").on("click", function() {

  // Button click confirmation
  // console.log("Submitted..."); 

  // Initialise model input object
  var input = {};

  // Grab plot data and append to object
  var plot = d3.select("#inputPlot").property("value");  
  
  // Check if plot is null
  if (!plot) {
    // If no plot is supplied, show the spinner indicating response but gives a notification after...
    d3.select(this)
    .html("<span class='spinner-border' role='status' aria-hidden='true'></span><h4 id='rate-buffer'>Predicting...</h4>")
    .classed("disabled", true);

    // Wait 1 second and then remove spinner
    setTimeout(() => {
      // Remove spinner
      d3.select("#submit-btn")
      .html("<h4 id='rate-buffer'>Rate Your Movie</h4>")
      .classed("disabled", false);

      // Also prompt the user to give it a plot
      d3.select("#plot p small")
      .text("Please enter a plot in the text box to get more accurate result :)")
      .classed("text-danger", true);
      // Erase margin on <p>
      d3.select("#plot p")
      .style("margin-bottom", 0);

    }, 1000);
  }

  // If plot is not null, collect input and send to flask app...
  else {
    d3.select("#plot p small")
    .text("")
    .classed("text-danger", false);
    // Erase margin on <p>
    d3.select("#plot p")
    .style("margin-bottom", 0);

    input["plot"] = plot;

    // Grab genres and append to object
    // If button is selected, give value of 1, else 0
    var genres = d3.selectAll(".btn-outline-light");
    genres.each(function() {

        if (d3.select(this).classed("selected")) {
            var genre = this.value;
            input[genre] = parseInt(1);
        }
        else {
            var genre = this.value;
            input[genre] = parseInt(0);
        }
    })

    // Add spinner to indicate our server is trying to predict the result when the button is clicked
    d3.select(this)
      .html("<span class='spinner-border' role='status' aria-hidden='true'></span><h4 id='rate-buffer'>Predicting...</h4>")
      .classed("disabled", true);

    // // Final input to POST
    // console.log(input);

    // -----------------------

    // -----------------------
    // Send inputs to server
    fetch('/predict', {

      // Declare what type of data we're sending
      headers: {
        'Content-Type': 'application/json'
      },
      // Specify the method (POST)
      method: 'POST',
      // A JSON payload
      body: JSON.stringify({
        input
      })
    }).then(function (response) { // At this point, Flask has printed our JSON
      // console.log(response);
      return response.json();
    }).then(function (prediction) {

      // // Present prediction
      // console.log(prediction);
      
      // Condition to categorize prediction outcome to HTML template
      if (prediction["class"] == 0) {
        d3.select("#rating")
        .text("G")
        .classed("text-success text-info text-warning text-danger", false) // remove class first before append new ones
        .classed("text-success", true);
        // Add probability
        d3.select("#probability")
        .text(`(Confidence: ${prediction["prob"]})`);
      }
      else if (prediction["class"] == 1) {
        d3.select("#rating")
        .text("PG")
        .classed("text-success text-info text-warning text-danger", false)
        .classed("text-info", true);
        // Add probability
        d3.select("#probability")
        .text(`(Confidence: ${prediction["prob"]})`);
      }
      else if (prediction["class"] == 2) {
        d3.select("#rating")
        .text("PG-13")
        .classed("text-success text-info text-warning text-danger", false)
        .classed("text-warning", true);
        // Add probability
        d3.select("#probability")
        .text(`(Confidence: ${prediction["prob"]})`);
      }
      else if (prediction["class"] == 3) { // Rated-R
        d3.select("#rating")
        .text("R")
        .classed("text-success text-info text-warning text-danger", false)
        .classed("text-danger", true);
        // Add probability
        d3.select("#probability")
        .text(`(Confidence: ${prediction["prob"]})`);
      }
      else { // Remove rating if no plot is entered
        d3.select("#rating")
        .text("")
        .classed("text-success text-info text-warning text-danger", false);
        // Remove probability
        d3.select("#probability")
        .text("");
      }

      // Restore the original button text and functionality after prediction outcome is returned
      d3.select("#submit-btn")
        .html("<h4 id='rate-buffer'>Rate Your Movie</h4>")
        .classed("disabled", false);
    });

  // else statement ends here...
  }
});





