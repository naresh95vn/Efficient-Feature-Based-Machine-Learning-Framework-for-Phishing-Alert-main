var url,b = 0,rank = new Array(),gRank;
chrome.runtime.onMessage.addListener(function(response, sender, sendResponse){
	url = sender.url;
	url = "https://www.alexa.com/siteinfo/"+url;
	$.get(url,function(data){
		$(data).find("strong").each(function(){
			if((this.innerText).match(/[0-9]+/g) != null){
				rank.push(this.innerText);
			}
		});
		if(rank[0] != undefined){
			var gRank = (parseInt((rank[0]).replace(/,/g,"")));
		}
		$(data).find("strong").each(function(){
			var str = this.innerText;
			if(str.match(/We don't have enough data to rank this website./g)){
				b = 1;
			}
		});
		if(( b == 1 || gRank > 100000 ) && (response == 1 || response == 2)){
			//chrome.runtime.sendMessage([response,b,gRank]);
			alert("This website might be a phishing website");
		}
		});
});