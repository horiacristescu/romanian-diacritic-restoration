$(function(){

	function post_success(data) {
		$("#text").val(data)
	}

    $("#btn").click(function(){
    	var txt = $("#text").val()
        $.ajax({
			type: "POST",
			url: "/ajax",
			data: txt,
			success: post_success,
			dataType: "text"
		});
    })
})
