
$('#form_demo').on("submit", function(event){
    event.preventDefault();
    $.ajax({ 
        url: "http://localhost:5000/api/v1/predict",
        method: "POST",
        crossDomain: true,
        data: new FormData(this),
        contentType: false,
        cache: false,
        processData: false,
        dataType: "json",
        

    })
    .then ((res) => {
        // res.data.path_file
        // let div= $('.rest');
        // div.style.background = `url(${res.data.label}) no-repeat`;


        alert("Kết quả là : "+res.data.label);
    })
    
})