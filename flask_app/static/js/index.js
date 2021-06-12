// Keep

// classes selected genres
d3.selectAll(".btn-outline-light").on("click", function(){
    var selection = d3.select(this);
    if (selection.classed("selected")){
      selection.classed("selected", false);
    }
    else{
      selection.classed("selected",true);
    };
});

//submission function
d3.select("#submit-btn").on("click", function(){

    // button click confirmation
    console.log("submit button clicked"); 

    // initialise model input object
    var input = {};

    // grab plot data and append to object
    var plot = d3.select("#inputPlot").property("value");  
    input["plot"] = plot;


    // grab genres and append to object 
    // if selection give value of 1, else 0
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

    // final input to POST
    console.log(input);


    //send input to serve
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




  
