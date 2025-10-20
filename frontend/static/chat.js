var bubble_container = '<div class="bubble_container"></div>';
var bubble = '<div class="bubble"></div>';
var loader = '<div id="loading_bar"><div id="circle1" class="circle"></div><div id="circle2" class="circle"></div><div id="circle3" class="circle"></div></div>';

var chat_history = [];
function enterText(){
    let text = $('#user_text').val();
    if(text.length == 0){
        return null;
    }
    $('#user_text').prop('disabled', true);
    let html_text = text.replace(/[\u00A0-\u9999<>\&]/gim, function(i) {
    return '&#' + i.charCodeAt(0) + ';';
    });
    let html = $(bubble).html(html_text);
    html = $(bubble_container).addClass('user').append(html);
    $('#chat_content').append(html);
    chat_history.push({'role':'user','message':text});
    loadResponse();
    $('#user_text').val('');
    window.setTimeout(function(){
        $("#chat_content").animate({ scrollTop: $('#chat_content')[0].scrollHeight+"px" }, 300);       
    }, 50);     
}

var new_chat = false;
var buffer = "";


  var loaders = {};
    var loaders_interval = null;

    function start_loader(loader_id){
      let num_loaders = Object.keys(loaders).length;
      $(loader_id).html('<div class="loading_bar"><div class="circle circle1"></div><div class="circle circle2"></div><div class="circle circle3"></div></div>');
      $(loader_id).show();
      loaders[loader_id] = 0;
      if(num_loaders == 0){
        loaders_interval = window.setInterval(loader_interval, 400);
      }
    }

    function loader_interval(){
      for(loader in loaders){
        let position = (loaders[loader] +1) % 3;
        $(loader+' .circle').removeClass('active');
        $(loader+' .circle:nth-child('+(position+1)+')').addClass('active');
        loaders[loader] = position;
      }        
    }

    function stop_loader(loader_id){
      $(loader_id).hide();
      $(loader_id).html('');
      delete loaders[loader_id];
      if(Object.keys(loaders).length == 0){
        window.clearInterval(loaders_interval);
      }
    }

  function loadResponse(){
    let html = $(bubble).html('<div class="chat-loader"></div>');
    html = $(bubble_container).addClass('bot').append(html);
    $('#chat_content').append(html);
    window.setTimeout(function(){ 
      start_loader('.chat-loader');
        $("#chat_content").animate({ scrollTop: $('#chat_content')[0].scrollHeight+"px" }, 300);          
    }, 50);
    
    buffer = "";
    fetch('/response', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: $.param({
        history: JSON.stringify(chat_history),
        doku: false
    })
}).then(response => {
    new_chat = true;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    function read() {
        reader.read().then(({ done, value }) => {
          console.log("read",done,value);
            if (done) {
                console.log("done"); 
                chat_history.push({ role: 'assistant', message: buffer });
                $('#user_text').prop('disabled', false);
                if ($(window).width() > 414) {
                    $('#user_text').focus();
                }
                return;
            }
    
            const chunk = decoder.decode(value, { stream: true });
            console.log("chunk",chunk);
            // Split falls mehrere Nachrichten in einem Chunk sind
            const parts = chunk.split("\f\f");
            var extra_started = false;
            var extra_text = "";
            for (const part of parts) {
                if (!part.trim()) continue;
                if(extra_started){
                  extra_text += jsonLine.substring(5);
                   try{
                    const extra = JSON.parse(extra_text);
                    parseExtra(extra)
                  }
                  catch{
                    console.log("still not ready")
                  }
                }
                else if (part.startsWith("data:")) {
                    const data = part.substring(6);
                    if (data !== "[start]") {
                        if (new_chat) {
                            buffer = data;
                            enterResponse(buffer, true);
                            new_chat = false;
                        } else {
                            buffer += data;
                            const html = parseMarkdown(buffer);
                            $('.bubble_container:last-child .bubble').html(html);
                        }
                    }
                } else if (part.startsWith("event: extra")) {
                    const jsonLine = part.split("\f").find(line => line.startsWith("data:"));
                    extra_started = true;

                    console.log("jsonLine", jsonLine);
                    if (jsonLine) {
                        try{
                          const extra = JSON.parse(jsonLine.substring(5));
                          parseExtra(extra)
                        }
                        catch{
                          extra_text = jsonLine.substring(5);
                        }
                       
                    }
                }
            }
    
            read(); // continue reading
        });
    }

    read();
});
}
    function parseExtra(extra){
       if (extra.linked_html) {
                            
                   
          $('.bubble_container:last-child .bubble').html(parseMarkdown(extra.linked_html));
      }
    }
    

function parseMarkdown(text){
  html = marked.parse(text).replace(/\n\t[ ]?\*/g,"<br/> * ");
  return html;
}
function enterResponse(text,stream=false){
    stop_loader('.chat-loader');
    // html_text = text.replace(/[\u00A0-\u9999<>\&]/gim, function(i) {
    //     return '&#' + i.charCodeAt(0) + ';';
    // });
    html_text = text;
    const html = parseMarkdown(text);
    $('.bubble_container:last-child .bubble').html(html);  
    if(!stream){
        chat_history.push({'role':'assistant','message':text});
        $('#user_text').prop('disabled', false);
        if($(window).width() > 414){
            $('#user_text').focus();
        }
    }
    
    window.setTimeout(function(){
        $("#chat_content").animate({ scrollTop: $('#chat_content')[0].scrollHeight+"px" }, 300);          
    }, 50);
}
$("#user_text").keypress(function (e) {
    if (e.which == 13) {
        enterText();
    }
});
$(document).ready(function(){
    $('#user_text').focus();
})
function nl2br (str, is_xhtml) {
    if (typeof str === 'undefined' || str === null) {
        return '';
    }
    var breakTag = (is_xhtml || typeof is_xhtml === 'undefined') ? '<br />' : '<br>';
    return (str + '').replace(/([^>\r\n]?)(\r\n|\n\r|\r|\n)/g, '$1' + breakTag + '$2');
}

