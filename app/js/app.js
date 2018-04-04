$(function(){

	function post_success(data) {
		$("#text").html(data)
	    $("#text").focus()
	}

    $("#btn").click(function(){
    	var txt = $("#text").text()
        $.ajax({
			type: "POST",
			url: "/ajax",
			data: txt,
			success: post_success,
			dataType: "text"
		});
    })

	$("[contenteditable]").focusout(function(){
        var element = $(this);        
        if (!element.text().trim().length) {
            element.empty();
        }
    });

})
