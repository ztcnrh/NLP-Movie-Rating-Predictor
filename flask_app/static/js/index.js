// Class selected genres & apply styling
d3.selectAll(".btn-outline-light").on("click", function(){
    var selection = d3.select(this);
    if (selection.classed("selected")){
      selection.classed("selected", false);
    }
    else{
      selection.classed("selected",true);
    };
});


// Submission function
d3.select("#submit-btn").on("click", function(){

    // Button click confirmation
    console.log("Submitted..."); 

    // Initialise model input object
    var input = {};

    // Grab plot data and append to object
    var plot = d3.select("#inputPlot").property("value");  
    
    // Check if plot is null. Notify the user if it is...
    if (!plot) {
        d3.select("#plot")
        .append("p")
        .style("margin-bottom", 0)
        .append("small")
        .text("Please enter some plot in the text box :)")
        .classed("text-danger", true)
    }

    input["plot"] = plot;


    // Grab genres and append to object 
    // Ff selection give value of 1, else 0
    var genres = d3.selectAll(".btn-outline-light");
    genres.each(function(){

        if(d3.select(this).classed("selected")){
            var genre = this.value;
            input[genre] = parseInt(1);
        }
        else{
            var genre = this.value;
            input[genre] = parseInt(0);
        }
    })

    // Final input to POST
    console.log(input);


    // Send input to server
    $.ajax({
      type: 'POST',
      url: '/predict',
      data: JSON.stringify(input),
      dataType: 'json',
      contentType: 'application/json; charset=utf-8'
      }).done(function(msg) {
      console.log(msg);
  });

})








// // Submit button stylings
// submitButton = d3.selectAll("submit-btn");

// submitButton.on("click", function() {
//     d3.select(this)
//     .style("btn-danger", true)
//     .style("background-color", "red")
// })
// .on("mouseout", function() {
//     d3.select(this)
//     .classed("btn-danger", false)
// });


