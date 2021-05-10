<!doctype html>
<html lang="fr">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css" integrity="sha384-B0vP5xmATw1+K9KRQjQERJvTumQW0nPEzvF6L/Z6nronJ3oUOFUFpCjEUQouq2+l" crossorigin="anonymous">

    <title>Résultats formidrive</title>
  </head>
  <body>
    <h1>Résultats</h1>
<div class="container">
<div class="row">

<?php
foreach(glob("./".'*') as $filename){
$ext = pathinfo($filename, PATHINFO_EXTENSION);
if($ext=='jpg' || $ext=='png')
{
echo "<a class='col-3' href='$filename' >";
echo "<img class='img-fluid' src='$filename' />";
echo "</a>";
}
}
?>
</div>
</div>
  </body>
</html>
