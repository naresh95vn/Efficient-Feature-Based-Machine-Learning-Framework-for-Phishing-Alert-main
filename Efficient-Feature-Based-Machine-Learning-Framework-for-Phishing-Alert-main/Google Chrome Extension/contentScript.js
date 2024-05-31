var a = 0;
$("a").each(function(){
  if((this.href).matches(/javascript:void(0)/g) || (this.target).matches(/_blank/g)){
	  a=1;
  }
});
var b;
if(window.location.protocol != "https:"){
	b=a+1;
}
chrome.runtime.sendMessage(b);



