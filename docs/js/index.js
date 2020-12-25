$(function(){
  $('#tabs').tabs().on('click', 'a', function(e) {
     switch(e.target.id) {
      case 'language':
      case 'forum':
         location.href = e.target.href;
     }
  });

  $("#tab_classical").tabs();
  $("#tab_quantum").tabs();
  $("#tab_quantum_energy").tabs();
});
